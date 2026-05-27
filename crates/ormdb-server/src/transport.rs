//! Server transport layer using async-nng.
//!
//! Provides TCP and IPC transport for the ORMDB server using NNG's REP socket.
//! Supports TLS encryption, rate limiting, and connection limits.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, RwLock};
use std::thread;
use std::time::{Duration, Instant};

use async_nng::AsyncContext;
use nng::options::Options;
use nng::{Message, Protocol, Socket};

use ormdb_proto::framing::encode_frame;
use ormdb_proto::{error_codes, Request, Response};

use crate::config::{RateLimitConfig, ServerConfig};
use crate::error::Error;
use crate::handler::RequestHandler;

/// Transport metrics for monitoring.
#[derive(Debug)]
pub struct TransportMetrics {
    /// Total number of requests received.
    pub requests_total: AtomicU64,
    /// Number of successful requests.
    pub requests_success: AtomicU64,
    /// Number of failed requests.
    pub requests_failed: AtomicU64,
    /// Number of bytes received.
    pub bytes_received: AtomicU64,
    /// Number of bytes sent.
    pub bytes_sent: AtomicU64,
    /// Server start time.
    pub started_at: Instant,
}

impl TransportMetrics {
    /// Create new metrics.
    fn new() -> Self {
        Self {
            requests_total: AtomicU64::new(0),
            requests_success: AtomicU64::new(0),
            requests_failed: AtomicU64::new(0),
            bytes_received: AtomicU64::new(0),
            bytes_sent: AtomicU64::new(0),
            started_at: Instant::now(),
        }
    }

    /// Record a successful request.
    fn record_success(&self, received_bytes: usize, sent_bytes: usize) {
        self.requests_total.fetch_add(1, Ordering::Relaxed);
        self.requests_success.fetch_add(1, Ordering::Relaxed);
        self.bytes_received.fetch_add(received_bytes as u64, Ordering::Relaxed);
        self.bytes_sent.fetch_add(sent_bytes as u64, Ordering::Relaxed);
    }

    /// Record a failed request.
    fn record_failure(&self, received_bytes: usize, sent_bytes: usize) {
        self.requests_total.fetch_add(1, Ordering::Relaxed);
        self.requests_failed.fetch_add(1, Ordering::Relaxed);
        self.bytes_received.fetch_add(received_bytes as u64, Ordering::Relaxed);
        self.bytes_sent.fetch_add(sent_bytes as u64, Ordering::Relaxed);
    }

    /// Get the uptime duration.
    pub fn uptime(&self) -> Duration {
        self.started_at.elapsed()
    }

    /// Get total requests count.
    pub fn total_requests(&self) -> u64 {
        self.requests_total.load(Ordering::Relaxed)
    }

    /// Get successful requests count.
    pub fn successful_requests(&self) -> u64 {
        self.requests_success.load(Ordering::Relaxed)
    }

    /// Get failed requests count.
    pub fn failed_requests(&self) -> u64 {
        self.requests_failed.load(Ordering::Relaxed)
    }

    /// Get total bytes received.
    pub fn total_bytes_received(&self) -> u64 {
        self.bytes_received.load(Ordering::Relaxed)
    }

    /// Get total bytes sent.
    pub fn total_bytes_sent(&self) -> u64 {
        self.bytes_sent.load(Ordering::Relaxed)
    }
}

impl Default for TransportMetrics {
    fn default() -> Self {
        Self::new()
    }
}

/// Token bucket rate limiter.
#[derive(Debug)]
pub struct RateLimiter {
    /// Tokens available.
    tokens: AtomicU64,
    /// Maximum tokens (burst size).
    max_tokens: u64,
    /// Tokens added per second.
    refill_rate: u64,
    /// Last refill time.
    last_refill: RwLock<Instant>,
    /// Whether rate limiting is enabled.
    enabled: bool,
}

impl RateLimiter {
    /// Create a new rate limiter.
    pub fn new(config: &RateLimitConfig) -> Self {
        Self {
            tokens: AtomicU64::new(config.burst_size as u64),
            max_tokens: config.burst_size as u64,
            refill_rate: config.requests_per_second as u64,
            last_refill: RwLock::new(Instant::now()),
            enabled: config.enabled,
        }
    }

    /// Create a disabled rate limiter.
    pub fn disabled() -> Self {
        Self {
            tokens: AtomicU64::new(u64::MAX),
            max_tokens: u64::MAX,
            refill_rate: 0,
            last_refill: RwLock::new(Instant::now()),
            enabled: false,
        }
    }

    /// Try to acquire a token. Returns true if request is allowed.
    pub fn try_acquire(&self) -> bool {
        if !self.enabled {
            return true;
        }

        // Refill tokens based on elapsed time
        self.refill();

        // Try to decrement token count
        loop {
            let current = self.tokens.load(Ordering::Relaxed);
            if current == 0 {
                return false;
            }
            if self
                .tokens
                .compare_exchange_weak(current, current - 1, Ordering::Relaxed, Ordering::Relaxed)
                .is_ok()
            {
                return true;
            }
        }
    }

    /// Refill tokens based on elapsed time.
    fn refill(&self) {
        let mut last_refill = self.last_refill.write().unwrap();
        let now = Instant::now();
        let elapsed = now.duration_since(*last_refill);
        let elapsed_secs = elapsed.as_secs_f64();

        if elapsed_secs >= 0.001 {
            // Only refill every millisecond minimum
            let new_tokens = (elapsed_secs * self.refill_rate as f64) as u64;
            if new_tokens > 0 {
                let current = self.tokens.load(Ordering::Relaxed);
                let new_total = (current + new_tokens).min(self.max_tokens);
                self.tokens.store(new_total, Ordering::Relaxed);
                *last_refill = now;
            }
        }
    }
}

/// Connection tracker for limiting concurrent connections.
#[derive(Debug)]
pub struct ConnectionTracker {
    /// Current active connections.
    active: AtomicU64,
    /// Maximum allowed connections.
    max: u64,
    /// Whether connection limiting is enabled.
    enabled: bool,
    /// Rate limited requests counter.
    rate_limited: AtomicU64,
    /// Connection limit rejections counter.
    connection_rejected: AtomicU64,
}

impl ConnectionTracker {
    /// Create a new connection tracker.
    pub fn new(max_connections: u32, enabled: bool) -> Self {
        Self {
            active: AtomicU64::new(0),
            max: max_connections as u64,
            enabled,
            rate_limited: AtomicU64::new(0),
            connection_rejected: AtomicU64::new(0),
        }
    }

    /// Try to acquire a connection slot. Returns true if allowed.
    pub fn try_acquire(&self) -> bool {
        if !self.enabled {
            return true;
        }

        loop {
            let current = self.active.load(Ordering::Relaxed);
            if current >= self.max {
                self.connection_rejected.fetch_add(1, Ordering::Relaxed);
                return false;
            }
            if self
                .active
                .compare_exchange_weak(current, current + 1, Ordering::Relaxed, Ordering::Relaxed)
                .is_ok()
            {
                return true;
            }
        }
    }

    /// Release a connection slot.
    pub fn release(&self) {
        if self.enabled {
            self.active.fetch_sub(1, Ordering::Relaxed);
        }
    }

    /// Record a rate-limited request.
    pub fn record_rate_limited(&self) {
        self.rate_limited.fetch_add(1, Ordering::Relaxed);
    }

    /// Get the count of rate-limited requests.
    pub fn rate_limited_count(&self) -> u64 {
        self.rate_limited.load(Ordering::Relaxed)
    }

    /// Get the count of connection rejections.
    pub fn connection_rejected_count(&self) -> u64 {
        self.connection_rejected.load(Ordering::Relaxed)
    }

    /// Get current active connections.
    pub fn active_connections(&self) -> u64 {
        self.active.load(Ordering::Relaxed)
    }
}

/// Server transport that handles incoming connections.
pub struct Transport {
    socket: Socket,
    handler: Arc<RequestHandler>,
    max_message_size: usize,
    metrics: Arc<TransportMetrics>,
    request_timeout: Duration,
    worker_count: usize,
    rate_limiter: Arc<RateLimiter>,
    connection_tracker: Arc<ConnectionTracker>,
    tls_enabled: bool,
}

impl Transport {
    /// Create a new transport with the given configuration and request handler.
    pub fn new(config: &ServerConfig, handler: Arc<RequestHandler>) -> Result<Self, Error> {
        // Create REP socket
        let socket = Socket::new(Protocol::Rep0)
            .map_err(|e| Error::Transport(format!("failed to create socket: {}", e)))?;

        // Set socket options
        socket
            .set_opt::<nng::options::RecvMaxSize>(config.max_message_size)
            .map_err(|e| Error::Transport(format!("failed to set max message size: {}", e)))?;

        // Configure TLS if enabled
        let tls_enabled = config.tls.enabled;
        if tls_enabled {
            // Set TLS certificate and key
            if let (Some(cert_path), Some(key_path)) =
                (&config.tls.cert_path, &config.tls.key_path)
            {
                let cert_key_path = format!(
                    "{}:{}",
                    cert_path.display(),
                    key_path.display()
                );
                socket
                    .set_opt::<nng::options::transport::tls::CertKeyFile>(cert_key_path)
                    .map_err(|e| {
                        Error::Transport(format!("failed to set TLS certificate/key: {}", e))
                    })?;

                tracing::info!(
                    cert = %cert_path.display(),
                    key = %key_path.display(),
                    "TLS configured"
                );
            } else {
                return Err(Error::Config(
                    "TLS enabled but certificate or key path not provided".to_string(),
                ));
            }

            // Set CA certificate if provided (for client verification)
            if let Some(ca_path) = &config.tls.ca_path {
                socket
                    .set_opt::<nng::options::transport::tls::CaFile>(ca_path.display().to_string())
                    .map_err(|e| Error::Transport(format!("failed to set TLS CA: {}", e)))?;

                tracing::info!(ca = %ca_path.display(), "TLS CA certificate configured");
            }
        }

        // Bind to TCP address if configured
        if let Some(tcp_addr) = &config.tcp_address {
            // Use tls+tcp:// scheme if TLS is enabled
            let listen_addr = if tls_enabled {
                tcp_addr.replace("tcp://", "tls+tcp://")
            } else {
                tcp_addr.clone()
            };

            socket
                .listen(&listen_addr)
                .map_err(|e| Error::Transport(format!("failed to listen on {}: {}", listen_addr, e)))?;

            tracing::info!(
                address = %listen_addr,
                tls = tls_enabled,
                "listening on TCP"
            );
        }

        // Bind to IPC address if configured (IPC doesn't support TLS)
        if let Some(ipc_addr) = &config.ipc_address {
            socket
                .listen(ipc_addr)
                .map_err(|e| Error::Transport(format!("failed to listen on {}: {}", ipc_addr, e)))?;

            tracing::info!(address = %ipc_addr, "listening on IPC");
        }

        // Create rate limiter
        let rate_limiter = Arc::new(RateLimiter::new(&config.rate_limit));
        if config.rate_limit.enabled {
            tracing::info!(
                rps = config.rate_limit.requests_per_second,
                burst = config.rate_limit.burst_size,
                "rate limiting enabled"
            );
        }

        // Create connection tracker
        let connection_tracker = Arc::new(ConnectionTracker::new(
            config.connection_limits.max_connections,
            config.connection_limits.enabled,
        ));
        if config.connection_limits.enabled {
            tracing::info!(
                max = config.connection_limits.max_connections,
                "connection limits enabled"
            );
        }

        Ok(Self {
            socket,
            handler,
            max_message_size: config.max_message_size,
            metrics: Arc::new(TransportMetrics::new()),
            request_timeout: config.request_timeout,
            worker_count: config.transport_workers.max(1),
            rate_limiter,
            connection_tracker,
            tls_enabled,
        })
    }

    /// Get a reference to the transport metrics.
    pub fn metrics(&self) -> &TransportMetrics {
        &self.metrics
    }

    /// Run the transport loop, processing incoming requests.
    pub async fn run(&self) -> Result<(), Error> {
        let stop_flag = Arc::new(AtomicBool::new(false));
        let _handles = self.spawn_worker_threads(stop_flag)?;

        tracing::info!("transport ready, accepting requests");
        std::future::pending::<()>().await;
        Ok(())
    }

    /// Run the transport with graceful shutdown support.
    pub async fn run_until_shutdown(
        &self,
        mut shutdown: tokio::sync::broadcast::Receiver<()>,
    ) -> Result<(), Error> {
        let stop_flag = Arc::new(AtomicBool::new(false));
        let handles = self.spawn_worker_threads(stop_flag.clone())?;

        tracing::info!("transport ready, accepting requests");

        let _ = shutdown.recv().await;
        tracing::info!(
            total_requests = self.metrics.total_requests(),
            successful = self.metrics.successful_requests(),
            failed = self.metrics.failed_requests(),
            bytes_received = self.metrics.total_bytes_received(),
            bytes_sent = self.metrics.total_bytes_sent(),
            uptime_secs = self.metrics.uptime().as_secs(),
            "shutdown signal received, stopping transport"
        );

        stop_flag.store(true, Ordering::SeqCst);
        let _ = tokio::task::spawn_blocking(move || {
            for handle in handles {
                let _ = handle.join();
            }
        })
        .await;

        Ok(())
    }

    /// Process a raw message and return the response bytes.
    fn process_message(&self, data: &[u8]) -> Vec<u8> {
        self.worker().process_message_with_status(data).0
    }

    /// Process a raw message and return (response bytes, is_success).
    fn process_message_with_status(&self, data: &[u8]) -> (Vec<u8>, bool) {
        self.worker().process_message_with_status(data)
    }

    fn worker(&self) -> TransportWorker {
        TransportWorker::new(
            self.handler.clone(),
            self.max_message_size,
            self.rate_limiter.clone(),
            self.connection_tracker.clone(),
        )
    }

    /// Get connection tracker statistics.
    pub fn connection_stats(&self) -> (u64, u64, u64) {
        (
            self.connection_tracker.active_connections(),
            self.connection_tracker.rate_limited_count(),
            self.connection_tracker.connection_rejected_count(),
        )
    }

    /// Check if TLS is enabled.
    pub fn is_tls_enabled(&self) -> bool {
        self.tls_enabled
    }

    fn spawn_worker_threads(
        &self,
        stop_flag: Arc<AtomicBool>,
    ) -> Result<Vec<thread::JoinHandle<()>>, Error> {
        let mut handles = Vec::with_capacity(self.worker_count);
        for worker_id in 0..self.worker_count {
            let socket = self.socket.clone();
            let worker = self.worker();
            let metrics = self.metrics.clone();
            let request_timeout = self.request_timeout;
            let stop_flag = stop_flag.clone();

            let handle = thread::Builder::new()
                .name(format!("ormdb-transport-{}", worker_id))
                .spawn(move || {
                    let runtime = tokio::runtime::Builder::new_current_thread()
                        .enable_all()
                        .build()
                        .expect("failed to build transport worker runtime");

                    runtime.block_on(async move {
                        let mut ctx = match AsyncContext::try_from(&socket) {
                            Ok(ctx) => ctx,
                            Err(e) => {
                                tracing::error!(error = %e, worker_id, "failed to create async context");
                                return;
                            }
                        };

                        loop {
                            if stop_flag.load(Ordering::SeqCst) {
                                tracing::info!(worker_id, "transport worker stopping");
                                return;
                            }

                            match ctx.receive(Some(Duration::from_secs(1))).await {
                                Ok(msg) => {
                                    let received_bytes = msg.len();
                                    let start = Instant::now();

                                    // Process request with timeout enforcement
                                    let process_result = tokio::time::timeout(
                                        request_timeout,
                                        tokio::task::spawn_blocking({
                                            let worker = worker.clone();
                                            let data = msg.as_slice().to_vec();
                                            move || worker.process_message_with_status(&data)
                                        })
                                    ).await;

                                    let (response_bytes, is_success) = match process_result {
                                        Ok(Ok((bytes, success))) => (bytes, success),
                                        Ok(Err(e)) => {
                                            // Task panic
                                            tracing::error!(error = %e, worker_id, "request processing panic");
                                            let response = Response::error(
                                                0,
                                                error_codes::INTERNAL,
                                                "internal processing error",
                                            );
                                            let bytes = worker.encode_response(&response)
                                                .unwrap_or_else(|_| worker.encode_minimal_error("processing error"));
                                            (bytes, false)
                                        }
                                        Err(_) => {
                                            // Timeout exceeded
                                            tracing::warn!(
                                                worker_id,
                                                timeout_ms = request_timeout.as_millis() as u64,
                                                "request timed out"
                                            );
                                            let response = Response::error(
                                                0,
                                                error_codes::TIMEOUT,
                                                format!("request timed out after {}ms", request_timeout.as_millis()),
                                            );
                                            let bytes = worker.encode_response(&response)
                                                .unwrap_or_else(|_| worker.encode_minimal_error("request timeout"));
                                            (bytes, false)
                                        }
                                    };

                                    let elapsed = start.elapsed();
                                    let sent_bytes = response_bytes.len();

                                    let response_msg = Message::from(response_bytes.as_slice());

                                    if let Err((_, e)) = ctx.send(response_msg, None).await {
                                        tracing::error!(error = %e, worker_id, "failed to send response");
                                        metrics.record_failure(received_bytes, 0);
                                    } else if is_success {
                                        metrics.record_success(received_bytes, sent_bytes);
                                    } else {
                                        metrics.record_failure(received_bytes, sent_bytes);
                                    }

                                    // Log slow requests (even if they didn't timeout)
                                    if elapsed > request_timeout / 2 {
                                        tracing::debug!(
                                            worker_id,
                                            duration_ms = elapsed.as_millis() as u64,
                                            timeout_ms = request_timeout.as_millis() as u64,
                                            "slow request"
                                        );
                                    }
                                }
                                Err(nng::Error::TimedOut) => {
                                    continue;
                                }
                                Err(e) => {
                                    tracing::error!(error = %e, worker_id, "receive error");
                                }
                            }
                        }
                    });
                })
                .map_err(|e| Error::Transport(format!("failed to spawn transport worker: {}", e)))?;

            handles.push(handle);
        }

        Ok(handles)
    }
}

#[derive(Clone)]
struct TransportWorker {
    handler: Arc<RequestHandler>,
    max_message_size: usize,
    rate_limiter: Arc<RateLimiter>,
    connection_tracker: Arc<ConnectionTracker>,
}

impl TransportWorker {
    fn new(
        handler: Arc<RequestHandler>,
        max_message_size: usize,
        rate_limiter: Arc<RateLimiter>,
        connection_tracker: Arc<ConnectionTracker>,
    ) -> Self {
        Self {
            handler,
            max_message_size,
            rate_limiter,
            connection_tracker,
        }
    }

    /// Process a raw message and return (response bytes, is_success).
    fn process_message_with_status(&self, data: &[u8]) -> (Vec<u8>, bool) {
        // Check rate limit
        if !self.rate_limiter.try_acquire() {
            self.connection_tracker.record_rate_limited();
            let response = Response::error(
                0,
                error_codes::BUDGET_EXCEEDED,
                "rate limit exceeded",
            );
            let bytes = self.encode_response(&response).unwrap_or_else(|_| {
                self.encode_minimal_error("rate limit exceeded")
            });
            return (bytes, false);
        }

        // Check connection limit
        if !self.connection_tracker.try_acquire() {
            let response = Response::error(
                0,
                error_codes::BUDGET_EXCEEDED,
                "connection limit exceeded",
            );
            let bytes = self.encode_response(&response).unwrap_or_else(|_| {
                self.encode_minimal_error("connection limit exceeded")
            });
            return (bytes, false);
        }

        // Process request
        let result = self.process_inner(data);

        // Release connection slot
        self.connection_tracker.release();

        result
    }

    /// Inner processing logic.
    fn process_inner(&self, data: &[u8]) -> (Vec<u8>, bool) {
        // Decode and process the request
        let (response, is_success) = match self.decode_and_handle(data) {
            Ok(response) => {
                let is_ok = response.status.is_ok();
                (response, is_ok)
            }
            Err(e) => {
                // A connection's first message is a Handshake, not a Request, so it
                // won't deserialize as one. Try handshake handling before erroring.
                if let Some(bytes) = self.try_handshake(data) {
                    return (bytes, true);
                }
                tracing::error!(error = %e, "request processing error");
                // Return error response with request ID 0 (unknown)
                let response = Response::error(0, error_codes::INTERNAL, e.to_string());
                (response, false)
            }
        };

        // Serialize response
        let bytes = match self.encode_response(&response) {
            Ok(bytes) => bytes,
            Err(e) => {
                tracing::error!(error = %e, "failed to encode response");
                // Try to send a minimal error response
                self.encode_minimal_error(&e.to_string())
            }
        };

        (bytes, is_success)
    }

    /// Try to handle the message as a connection handshake.
    ///
    /// The client sends a `Handshake` as the first message on a connection; the
    /// server has no separate handshake step, so we detect it here (only after a
    /// `Request` deserialize has failed, which keeps false positives away from
    /// real requests) and reply with a framed `HandshakeResponse`.
    fn try_handshake(&self, data: &[u8]) -> Option<Vec<u8>> {
        let payload = ormdb_proto::framing::extract_payload(data).ok()?;
        let mut aligned: rkyv::util::AlignedVec<16> = rkyv::util::AlignedVec::new();
        aligned.extend_from_slice(payload);
        let handshake =
            rkyv::from_bytes::<ormdb_proto::handshake::Handshake, rkyv::rancor::Error>(&aligned)
                .ok()?;

        let response = if ormdb_proto::handshake::is_version_compatible(
            handshake.protocol_version,
            ormdb_proto::PROTOCOL_VERSION,
        ) {
            ormdb_proto::handshake::HandshakeResponse::accept(
                ormdb_proto::PROTOCOL_VERSION,
                self.handler.schema_version(),
                "ormdb-server",
            )
        } else {
            ormdb_proto::handshake::HandshakeResponse::reject(format!(
                "incompatible protocol version: client={} server={}",
                handshake.protocol_version,
                ormdb_proto::PROTOCOL_VERSION
            ))
        };

        let payload = rkyv::to_bytes::<rkyv::rancor::Error>(&response).ok()?;
        encode_frame(&payload).ok()
    }

    /// Decode a request and dispatch to handler.
    fn decode_and_handle(&self, data: &[u8]) -> Result<Response, Error> {
        // Check message size
        if data.len() > self.max_message_size {
            return Err(Error::Protocol(ormdb_proto::Error::InvalidMessage(format!(
                "message too large: {} bytes (max: {})",
                data.len(),
                self.max_message_size
            ))));
        }

        // Extract payload from framed message
        let payload = ormdb_proto::framing::extract_payload(data)?;

        // Copy to aligned buffer for rkyv (required for zero-copy access)
        let mut aligned: rkyv::util::AlignedVec<16> = rkyv::util::AlignedVec::new();
        aligned.extend_from_slice(payload);

        // Deserialize request using rkyv
        let request: Request =
            rkyv::from_bytes::<Request, rkyv::rancor::Error>(&aligned).map_err(|e| {
                Error::Protocol(ormdb_proto::Error::InvalidMessage(format!(
                    "failed to deserialize request: {}",
                    e
                )))
            })?;

        // Handle the request
        Ok(self.handler.handle(&request))
    }

    /// Encode a response to framed bytes.
    pub(crate) fn encode_response(&self, response: &Response) -> Result<Vec<u8>, Error> {
        let payload = rkyv::to_bytes::<rkyv::rancor::Error>(response).map_err(|e| {
            Error::Protocol(ormdb_proto::Error::Serialization(format!(
                "failed to serialize response: {}",
                e
            )))
        })?;

        encode_frame(&payload).map_err(|e| Error::Protocol(e))
    }

    /// Create a minimal error response when normal encoding fails.
    pub(crate) fn encode_minimal_error(&self, message: &str) -> Vec<u8> {
        let response = Response::error(0, ormdb_proto::error_codes::INTERNAL, message);

        // Try to encode, fall back to empty on failure
        match rkyv::to_bytes::<rkyv::rancor::Error>(&response) {
            Ok(payload) => match encode_frame(&payload) {
                Ok(framed) => framed,
                Err(_) => Vec::new(),
            },
            Err(_) => Vec::new(),
        }
    }
}


/// Create a transport that listens on the configured addresses.
pub fn create_transport(
    config: &ServerConfig,
    handler: Arc<RequestHandler>,
) -> Result<Transport, Error> {
    if !config.has_transport() {
        return Err(Error::Config(
            "no transport configured (need TCP or IPC address)".to_string(),
        ));
    }

    Transport::new(config, handler)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::database::Database;
    use ormdb_core::catalog::{EntityDef, FieldDef, FieldType, ScalarType, SchemaBundle};
    use ormdb_proto::framing::MAX_MESSAGE_SIZE;

    fn setup_test_components() -> (tempfile::TempDir, Arc<RequestHandler>) {
        let dir = tempfile::tempdir().unwrap();
        let db = Database::open(dir.path()).unwrap();

        // Create schema
        let schema = SchemaBundle::new(1).with_entity(
            EntityDef::new("User", "id")
                .with_field(FieldDef::new("id", FieldType::Scalar(ScalarType::Uuid)))
                .with_field(FieldDef::new("name", FieldType::Scalar(ScalarType::String))),
        );
        db.catalog().apply_schema(schema).unwrap();

        let handler = Arc::new(RequestHandler::new(Arc::new(db)));
        (dir, handler)
    }

    #[test]
    fn test_transport_creation() {
        let (dir, handler) = setup_test_components();

        let ipc_path = format!("ipc://{}", dir.path().join("ormdb.sock").display());
        let config = ServerConfig::new(dir.path())
            .without_tcp()
            .with_ipc_address(ipc_path)
            .with_max_message_size(MAX_MESSAGE_SIZE);

        let transport = Transport::new(&config, handler);
        match transport {
            Ok(_) => {}
            Err(Error::Transport(msg)) if msg.contains("Permission denied") => {
                return;
            }
            Err(err) => panic!("transport creation failed: {err}"),
        }
    }

    #[test]
    fn test_transport_requires_address() {
        let (_dir, handler) = setup_test_components();

        let config = ServerConfig::new("/tmp/test").without_tcp();

        let result = create_transport(&config, handler);
        assert!(result.is_err());
    }

    fn create_test_worker(handler: Arc<RequestHandler>) -> TransportWorker {
        let rate_limiter = Arc::new(RateLimiter::disabled());
        let connection_tracker = Arc::new(ConnectionTracker::new(10000, false));
        TransportWorker::new(handler, MAX_MESSAGE_SIZE, rate_limiter, connection_tracker)
    }

    #[test]
    fn test_process_ping_message() {
        let (_dir, handler) = setup_test_components();
        let worker = create_test_worker(handler);

        // Create a ping request
        let request = Request::ping(42);
        let payload = rkyv::to_bytes::<rkyv::rancor::Error>(&request).unwrap();
        let framed = encode_frame(&payload).unwrap();

        // Process it
        let (response_bytes, is_success) = worker.process_message_with_status(&framed);
        assert!(is_success);

        // Decode response - copy to aligned buffer for rkyv
        let response_payload = ormdb_proto::framing::extract_payload(&response_bytes).unwrap();
        let mut aligned: rkyv::util::AlignedVec<16> = rkyv::util::AlignedVec::new();
        aligned.extend_from_slice(response_payload);
        let response: Response =
            rkyv::from_bytes::<Response, rkyv::rancor::Error>(&aligned).unwrap();

        assert_eq!(response.id, 42);
        assert!(response.status.is_ok());
        assert!(matches!(
            response.payload,
            ormdb_proto::ResponsePayload::Pong
        ));
    }

    #[test]
    fn test_process_invalid_message() {
        let (_dir, handler) = setup_test_components();
        let worker = create_test_worker(handler);

        // Send garbage data
        let (response_bytes, is_success) = worker.process_message_with_status(b"invalid data");

        // Should return an error response
        assert!(!response_bytes.is_empty());
        assert!(!is_success);
    }

    #[test]
    fn test_process_messages_concurrently() {
        let (_dir, handler) = setup_test_components();

        let mut handles = Vec::new();
        for i in 0..8 {
            let handler = handler.clone();
            handles.push(std::thread::spawn(move || {
                let worker = create_test_worker(handler);
                let request_id = 100 + i as u64;
                let request = Request::ping(request_id);
                let payload = rkyv::to_bytes::<rkyv::rancor::Error>(&request).unwrap();
                let framed = encode_frame(&payload).unwrap();

                let (response_bytes, is_success) = worker.process_message_with_status(&framed);
                assert!(is_success);

                let response_payload = ormdb_proto::framing::extract_payload(&response_bytes).unwrap();
                let mut aligned: rkyv::util::AlignedVec<16> = rkyv::util::AlignedVec::new();
                aligned.extend_from_slice(response_payload);
                let response: Response =
                    rkyv::from_bytes::<Response, rkyv::rancor::Error>(&aligned).unwrap();

                assert_eq!(response.id, request_id);
                assert!(response.status.is_ok());
            }));
        }

        for handle in handles {
            handle.join().unwrap();
        }
    }

    #[test]
    fn test_rate_limiting() {
        let (_dir, handler) = setup_test_components();

        // Create a rate limiter with very low limit
        let rate_config = RateLimitConfig::new(1, 1); // 1 RPS, 1 burst
        let rate_limiter = Arc::new(RateLimiter::new(&rate_config));
        let connection_tracker = Arc::new(ConnectionTracker::new(10000, false));
        let worker = TransportWorker::new(handler, MAX_MESSAGE_SIZE, rate_limiter, connection_tracker.clone());

        // First request should succeed
        let request = Request::ping(1);
        let payload = rkyv::to_bytes::<rkyv::rancor::Error>(&request).unwrap();
        let framed = encode_frame(&payload).unwrap();
        let (_, is_success) = worker.process_message_with_status(&framed);
        assert!(is_success);

        // Second immediate request should be rate limited
        let (response_bytes, is_success) = worker.process_message_with_status(&framed);
        assert!(!is_success);

        // Verify rate limited count
        assert!(connection_tracker.rate_limited_count() >= 1);
    }

    #[test]
    fn test_connection_limiting() {
        let (_dir, handler) = setup_test_components();

        // Create a connection tracker with limit of 1
        let rate_limiter = Arc::new(RateLimiter::disabled());
        let connection_tracker = Arc::new(ConnectionTracker::new(1, true));

        // Manually acquire the one available slot
        assert!(connection_tracker.try_acquire());

        // Create worker - it should fail to acquire
        let worker = TransportWorker::new(
            handler,
            MAX_MESSAGE_SIZE,
            rate_limiter,
            connection_tracker.clone(),
        );

        let request = Request::ping(1);
        let payload = rkyv::to_bytes::<rkyv::rancor::Error>(&request).unwrap();
        let framed = encode_frame(&payload).unwrap();

        // This should fail because connection limit is reached
        let (_, is_success) = worker.process_message_with_status(&framed);
        assert!(!is_success);
        assert!(connection_tracker.connection_rejected_count() >= 1);
    }
}
