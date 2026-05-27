//! Request handler for processing client requests.

use std::sync::Arc;
use std::time::{Duration, Instant};

use tracing::{debug, instrument, warn};

use ormdb_core::metrics::SharedMetricsRegistry;
use ormdb_core::query::{AggregateExecutor, ExplainService};
use ormdb_core::security::{
    AuditEvent, AuditLogger, CapabilityAuthenticator, DevAuthenticator, MutationOp,
    NullAuditLogger, SecurityContext, SecurityError,
};
use ormdb_proto::{
    error_codes, AggregateQuery, CacheMetrics, EntityCount, EntityQueryCount, MetricsResult,
    Mutation, MutationMetrics, MutationResult, Operation, QueryMetrics, ReplicationRole, ReplicationStatus,
    Request, Response, StorageMetrics, StreamChangesRequest, StreamChangesResponse,
    TransportMetrics,
};

#[cfg(feature = "raft")]
use ormdb_raft::{
    types::{ClientRequest, ClientResponse},
    RaftClusterManager,
};

use crate::database::Database;
use crate::error::Error;
use crate::mutation::MutationExecutor;
#[cfg(feature = "raft")]
use crate::mutation::{ensure_assigned_id, ensure_assigned_ids_batch};

const STATS_REFRESH_INTERVAL: Duration = Duration::from_secs(60);

/// Read consistency level applied to graph-fetch queries.
///
/// Lets a deployment select how reads observe concurrent writes — the axis the
/// object-graph isolation study compares.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ReadConsistency {
    /// Each sub-read sees the latest committed version independently (permits
    /// fractured graph reads). The historical default and fastest path.
    #[default]
    ReadCommitted,
    /// Root and all includes read as-of one commit watermark (graph-atomic).
    Snapshot,
    /// Snapshot read ordered with respect to the Raft log via ReadIndex
    /// (leader only). Falls back to `Snapshot` if Raft is not configured.
    Linearizable,
}

impl ReadConsistency {
    /// Parse from a string (e.g. an env var or CLI flag). Unknown values map to
    /// the default (read-committed).
    pub fn from_str_or_default(s: &str) -> Self {
        match s.trim().to_ascii_lowercase().as_str() {
            "snapshot" => ReadConsistency::Snapshot,
            "linearizable" | "linear" => ReadConsistency::Linearizable,
            _ => ReadConsistency::ReadCommitted,
        }
    }
}

/// Handles incoming requests and dispatches to appropriate handlers.
pub struct RequestHandler {
    database: Arc<Database>,
    metrics: Option<SharedMetricsRegistry>,
    authenticator: Arc<dyn CapabilityAuthenticator + Send + Sync>,
    audit_logger: Arc<dyn AuditLogger>,
    read_consistency: ReadConsistency,
    #[cfg(feature = "raft")]
    raft_manager: Option<Arc<RaftClusterManager>>,
}

impl RequestHandler {
    /// Create a new request handler with the given database.
    /// Uses the DefaultAuthenticator which grants all requested capabilities (development only).
    pub fn new(database: Arc<Database>) -> Self {
        Self {
            database,
            metrics: None,
            authenticator: Arc::new(DevAuthenticator),
            audit_logger: Arc::new(NullAuditLogger),
            read_consistency: ReadConsistency::default(),
            #[cfg(feature = "raft")]
            raft_manager: None,
        }
    }

    /// Create a new request handler with a custom authenticator.
    pub fn with_authenticator(
        database: Arc<Database>,
        authenticator: Arc<dyn CapabilityAuthenticator + Send + Sync>,
    ) -> Self {
        Self {
            database,
            metrics: None,
            authenticator,
            audit_logger: Arc::new(NullAuditLogger),
            read_consistency: ReadConsistency::default(),
            #[cfg(feature = "raft")]
            raft_manager: None,
        }
    }

    /// Create a new request handler with metrics support.
    /// Uses the DefaultAuthenticator which grants all requested capabilities (development only).
    pub fn with_metrics(database: Arc<Database>, metrics: SharedMetricsRegistry) -> Self {
        Self {
            database,
            metrics: Some(metrics),
            authenticator: Arc::new(DevAuthenticator),
            audit_logger: Arc::new(NullAuditLogger),
            read_consistency: ReadConsistency::default(),
            #[cfg(feature = "raft")]
            raft_manager: None,
        }
    }

    /// Create a new request handler with metrics and a custom authenticator.
    pub fn with_metrics_and_authenticator(
        database: Arc<Database>,
        metrics: SharedMetricsRegistry,
        authenticator: Arc<dyn CapabilityAuthenticator + Send + Sync>,
    ) -> Self {
        Self {
            database,
            metrics: Some(metrics),
            authenticator,
            audit_logger: Arc::new(NullAuditLogger),
            read_consistency: ReadConsistency::default(),
            #[cfg(feature = "raft")]
            raft_manager: None,
        }
    }

    /// Set the audit logger.
    pub fn with_audit_logger(mut self, logger: Arc<dyn AuditLogger>) -> Self {
        self.audit_logger = logger;
        self
    }

    /// Set the read consistency level applied to graph-fetch queries.
    pub fn with_read_consistency(mut self, mode: ReadConsistency) -> Self {
        self.read_consistency = mode;
        self
    }

    /// Create a new request handler with metrics and Raft support.
    #[cfg(feature = "raft")]
    pub fn with_metrics_and_raft(
        database: Arc<Database>,
        metrics: SharedMetricsRegistry,
        raft_manager: Option<Arc<RaftClusterManager>>,
    ) -> Self {
        Self {
            database,
            metrics: Some(metrics),
            authenticator: Arc::new(DevAuthenticator),
            audit_logger: Arc::new(NullAuditLogger),
            read_consistency: ReadConsistency::default(),
            raft_manager,
        }
    }

    /// Create a new request handler with metrics, Raft, and a custom authenticator.
    #[cfg(feature = "raft")]
    pub fn with_full_config(
        database: Arc<Database>,
        metrics: SharedMetricsRegistry,
        authenticator: Arc<dyn CapabilityAuthenticator + Send + Sync>,
        raft_manager: Option<Arc<RaftClusterManager>>,
    ) -> Self {
        Self {
            database,
            metrics: Some(metrics),
            authenticator,
            audit_logger: Arc::new(NullAuditLogger),
            read_consistency: ReadConsistency::default(),
            raft_manager,
        }
    }

    /// Handle a request and return a response.
    #[instrument(skip(self, request), fields(request_id = request.id, op = ?std::mem::discriminant(&request.operation)))]
    pub fn handle(&self, request: &Request) -> Response {
        let start = std::time::Instant::now();

        // Create security context from request credentials
        let context = match self.create_security_context(request) {
            Ok(ctx) => ctx,
            Err(e) => {
                // Log authentication failure
                let connection_id = format!("req-{}", request.id);
                let client_id = request.client_id.clone().unwrap_or_else(|| "anonymous".to_string());
                self.audit_logger.log(AuditEvent::authentication(
                    &connection_id,
                    &client_id,
                    false,
                    vec![],
                    Some(e.to_string()),
                ));
                return self.error_response(request.id, e);
            }
        };

        let result = self.handle_with_context(request, &context);

        let response = match result {
            Ok(response) => response,
            Err(e) => {
                // Log access denied events for security errors
                if let Error::Security(ref sec_err) = e {
                    use ormdb_core::security::SecurityError;
                    let (operation, entity) = self.extract_operation_info(request);
                    match sec_err {
                        SecurityError::PermissionDenied(_)
                        | SecurityError::CapabilityNotGranted(_)
                        | SecurityError::RlsViolation(_)
                        | SecurityError::FieldAccessDenied { .. } => {
                            self.audit_logger.log(AuditEvent::access_denied(
                                &context,
                                operation,
                                entity,
                                sec_err.to_string(),
                            ));
                        }
                        _ => {}
                    }
                }
                self.error_response(request.id, e)
            }
        };

        debug!(
            duration_us = start.elapsed().as_micros() as u64,
            success = response.status.is_ok(),
            client_id = %context.client_id,
            "request handled"
        );
        response
    }

    /// Extract operation and entity info from a request for audit logging.
    fn extract_operation_info(&self, request: &Request) -> (String, Option<String>) {
        match &request.operation {
            Operation::Query(q) => ("query".to_string(), Some(q.root_entity.clone())),
            Operation::Mutate(m) => {
                let op = match m {
                    Mutation::Insert { .. } => "insert",
                    Mutation::Update { .. } => "update",
                    Mutation::Upsert { .. } => "upsert",
                    Mutation::Delete { .. } => "delete",
                };
                (op.to_string(), Some(m.entity().to_string()))
            }
            Operation::MutateBatch(_) => ("mutate_batch".to_string(), None),
            Operation::GetSchema => ("get_schema".to_string(), None),
            Operation::Ping => ("ping".to_string(), None),
            Operation::Explain(q) => ("explain".to_string(), Some(q.root_entity.clone())),
            Operation::GetMetrics => ("get_metrics".to_string(), None),
            Operation::Aggregate(q) => ("aggregate".to_string(), Some(q.root_entity.clone())),
            Operation::Subscribe(_) => ("subscribe".to_string(), None),
            Operation::Unsubscribe { .. } => ("unsubscribe".to_string(), None),
            Operation::StreamChanges(_) => ("stream_changes".to_string(), None),
            Operation::GetReplicationStatus => ("get_replication_status".to_string(), None),
            Operation::ApplySchema(_) => ("apply_schema".to_string(), None),
        }
    }

    /// Create a security context from request credentials.
    fn create_security_context(&self, request: &Request) -> Result<SecurityContext, Error> {
        let connection_id = format!("req-{}", request.id);
        let client_id = request
            .client_id
            .clone()
            .unwrap_or_else(|| "anonymous".to_string());

        // Collect credentials into a vector for the authenticator
        let credentials: Vec<String> = request
            .credentials
            .as_ref()
            .map(|c| vec![c.clone()])
            .unwrap_or_default();

        let context = SecurityContext::from_handshake(
            connection_id.clone(),
            client_id.clone(),
            &credentials,
            self.authenticator.as_ref(),
        )?;

        // Log successful authentication
        let capabilities = context.capabilities.to_strings();
        self.audit_logger.log(AuditEvent::authentication(
            &connection_id,
            &client_id,
            true,
            capabilities,
            None,
        ));

        Ok(context)
    }

    /// Handle a request with security context.
    fn handle_with_context(
        &self,
        request: &Request,
        context: &SecurityContext,
    ) -> Result<Response, Error> {
        // Check schema version for operations that require it
        if matches!(
            request.operation,
            Operation::Query(_) | Operation::Mutate(_) | Operation::MutateBatch(_)
        ) {
            let server_version = self.database.schema_version();
            if request.schema_version != 0 && request.schema_version != server_version {
                return Ok(Response::error(
                    request.id,
                    error_codes::SCHEMA_MISMATCH,
                    format!(
                        "schema version mismatch: client has {}, server has {}",
                        request.schema_version, server_version
                    ),
                ));
            }
        }

        match &request.operation {
            Operation::Query(query) => {
                // Check read permission for the root entity
                context.require_read(&query.root_entity)?;
                // Check query depth against security budget
                self.check_query_budget(query, context)?;
                self.handle_query(request.id, query, context)
            }
            Operation::Mutate(mutation) => {
                // Check write/delete permission based on mutation type
                self.check_mutation_permission(context, mutation)?;
                self.handle_mutate(request.id, mutation, context)
            }
            Operation::MutateBatch(batch) => {
                // Check permissions for all mutations in the batch
                for mutation in &batch.mutations {
                    self.check_mutation_permission(context, mutation)?;
                }
                self.handle_batch(request.id, batch, context)
            }
            Operation::GetSchema => {
                // Schema read is allowed for all authenticated users
                self.handle_get_schema(request.id)
            }
            Operation::Ping => Ok(Response::pong(request.id)),
            Operation::Explain(query) => {
                // Explain requires read permission
                context.require_read(&query.root_entity)?;
                // Check query depth against security budget
                self.check_query_budget(query, context)?;
                self.handle_explain(request.id, query)
            }
            Operation::GetMetrics => {
                // Metrics require admin access
                context.require_admin()?;
                self.handle_get_metrics(request.id)
            }
            Operation::Aggregate(query) => {
                // Aggregate requires read permission
                context.require_read(&query.root_entity)?;
                self.handle_aggregate(request.id, query)
            }
            Operation::Subscribe(_) | Operation::Unsubscribe { .. } => {
                // Pub-sub operations require async handler integration (Phase 6)
                Ok(Response::error(
                    request.id,
                    error_codes::INVALID_REQUEST,
                    "pub-sub operations not yet available on this handler",
                ))
            }
            Operation::StreamChanges(req) => {
                // CDC/replication requires admin access
                context.require_admin()?;
                self.handle_stream_changes(request.id, req)
            }
            Operation::GetReplicationStatus => {
                // Replication status requires admin access
                context.require_admin()?;
                self.handle_replication_status(request.id)
            }
            Operation::ApplySchema(bytes) => {
                // Schema changes require admin access
                context.require_admin()?;
                self.handle_apply_schema(request.id, bytes)
            }
        }
    }

    /// Check mutation permissions based on mutation type.
    fn check_mutation_permission(
        &self,
        context: &SecurityContext,
        mutation: &Mutation,
    ) -> Result<(), Error> {
        let entity = mutation.entity();
        match mutation {
            Mutation::Insert { .. } | Mutation::Update { .. } | Mutation::Upsert { .. } => {
                context.require_write(entity)?;
            }
            Mutation::Delete { .. } => {
                context.require_delete(entity)?;
            }
        }
        Ok(())
    }

    /// Check query against security budget limits.
    fn check_query_budget(
        &self,
        query: &ormdb_proto::GraphQuery,
        context: &SecurityContext,
    ) -> Result<(), Error> {
        let budget = &context.budget;
        let query_depth = self.calculate_query_depth(query);

        if query_depth > budget.max_depth {
            return Err(Error::Security(SecurityError::BudgetExceeded(format!(
                "query depth {} exceeds limit {}",
                query_depth, budget.max_depth
            ))));
        }

        Ok(())
    }

    /// Calculate the maximum depth of a query based on its includes.
    fn calculate_query_depth(&self, query: &ormdb_proto::GraphQuery) -> usize {
        if query.includes.is_empty() {
            return 1;
        }

        let max_include_depth = query
            .includes
            .iter()
            .map(|inc| inc.depth())
            .max()
            .unwrap_or(0);

        max_include_depth + 1 // +1 for the root query
    }

    /// Handle a query operation.
    #[instrument(skip(self, query, context), fields(entity = %query.root_entity))]
    fn handle_query(
        &self,
        request_id: u64,
        query: &ormdb_proto::GraphQuery,
        context: &SecurityContext,
    ) -> Result<Response, Error> {
        let start = Instant::now();

        if let Err(e) = self
            .database
            .refresh_statistics_if_stale(STATS_REFRESH_INTERVAL)
        {
            warn!(error = %e, "Failed to refresh statistics");
        }

        // Create executor with security context for RLS and field masking
        let executor = if let Some(metrics) = &self.metrics {
            self.database.executor_with_metrics_and_security(metrics.clone(), context)
        } else {
            self.database.executor_with_security(context)
        };
        let result = match self.read_consistency {
            ReadConsistency::ReadCommitted => {
                let statistics = self.database.statistics();
                let cache = self.database.plan_cache();
                executor
                    .execute_with_cache(query, cache, Some(statistics))
                    .map_err(|e| Error::Database(format!("query execution failed: {}", e)))?
            }
            ReadConsistency::Snapshot => executor
                .execute_snapshot(query)
                .map_err(|e| Error::Database(format!("snapshot query failed: {}", e)))?,
            ReadConsistency::Linearizable => {
                // ReadIndex first (leader only); then a snapshot-consistent fetch.
                #[cfg(feature = "raft")]
                if let Some(raft) = &self.raft_manager {
                    raft.ensure_linearizable_blocking()
                        .map_err(|e| Error::Database(format!("linearizable read failed: {}", e)))?;
                }
                executor
                    .execute_snapshot(query)
                    .map_err(|e| Error::Database(format!("linearizable query failed: {}", e)))?
            }
        };

        let result_count = result.entities.first().map(|e| e.len()).unwrap_or(0);
        let duration_ms = start.elapsed().as_millis() as u64;

        // Log audit event
        self.audit_logger.log(AuditEvent::query(
            context,
            &query.root_entity,
            None, // TODO: add filter summary
            result_count,
            duration_ms,
        ));

        debug!(entities_returned = result_count, "query completed");
        Ok(Response::query_ok(request_id, result))
    }

    /// Handle an aggregate query operation.
    #[instrument(skip(self, query), fields(entity = %query.root_entity))]
    fn handle_aggregate(
        &self,
        request_id: u64,
        query: &AggregateQuery,
    ) -> Result<Response, Error> {
        let executor = AggregateExecutor::new(
            self.database.storage(),
            self.database.columnar(),
        );
        let result = executor
            .execute(query)
            .map_err(|e| Error::Database(format!("aggregate query failed: {}", e)))?;

        debug!(entity = %query.root_entity, aggregations = query.aggregations.len(), "aggregate query completed");
        Ok(Response::aggregate_ok(request_id, result))
    }

    /// Replicate a write through Raft and return the applied result.
    ///
    /// Only the leader accepts writes; followers reject with a hint to retry on
    /// the leader. The mutation is applied on every node via the state machine's
    /// apply callback (see `raft_apply::make_apply_fn`), so we do not also execute
    /// it locally here.
    #[cfg(feature = "raft")]
    fn raft_write(
        &self,
        raft: &RaftClusterManager,
        request: ClientRequest,
    ) -> Result<MutationResult, Error> {
        if !raft.is_leader() {
            return Err(Error::Database(
                "not the leader; writes must be sent to the current cluster leader".to_string(),
            ));
        }
        match raft
            .write_blocking(request)
            .map_err(|e| Error::Database(format!("raft write failed: {}", e)))?
        {
            ClientResponse::MutationResult(result) => Ok(result),
            ClientResponse::Error(msg) => Err(Error::Database(msg)),
            ClientResponse::NoopResult => {
                Err(Error::Database("unexpected noop response from raft".to_string()))
            }
        }
    }

    /// Handle a single mutation operation.
    #[instrument(skip(self, mutation, context), fields(entity = %mutation.entity(), mutation_type = ?std::mem::discriminant(mutation)))]
    fn handle_mutate(
        &self,
        request_id: u64,
        mutation: &ormdb_proto::Mutation,
        context: &SecurityContext,
    ) -> Result<Response, Error> {
        // In cluster mode, route the write through Raft consensus; the leader
        // applies it (and replicates to followers) via the apply callback.
        #[cfg(feature = "raft")]
        let result = if let Some(raft) = &self.raft_manager {
            // Assign the insert id on the leader so all nodes apply the same id.
            self.raft_write(raft, ClientRequest::Mutate(ensure_assigned_id(mutation)))?
        } else {
            MutationExecutor::new(&self.database).execute(mutation)?
        };
        #[cfg(not(feature = "raft"))]
        let result = MutationExecutor::new(&self.database).execute(mutation)?;

        // Determine mutation operation type
        let op = match mutation {
            Mutation::Insert { .. } => MutationOp::Insert,
            Mutation::Update { .. } => MutationOp::Update,
            Mutation::Upsert { .. } => MutationOp::Upsert,
            Mutation::Delete { .. } => MutationOp::Delete,
        };

        // Log audit event with affected entity IDs
        self.audit_logger.log(AuditEvent::mutation(
            context,
            mutation.entity(),
            op,
            result.inserted_ids.clone(),
        ));

        debug!(affected = result.affected, "mutation completed");
        Ok(Response::mutation_ok(request_id, result))
    }

    /// Handle a batch mutation operation.
    fn handle_batch(
        &self,
        request_id: u64,
        batch: &ormdb_proto::MutationBatch,
        context: &SecurityContext,
    ) -> Result<Response, Error> {
        #[cfg(feature = "raft")]
        let result = if let Some(raft) = &self.raft_manager {
            self.raft_write(raft, ClientRequest::MutateBatch(ensure_assigned_ids_batch(batch)))?
        } else {
            MutationExecutor::new(&self.database).execute_batch(batch)?
        };
        #[cfg(not(feature = "raft"))]
        let result = MutationExecutor::new(&self.database).execute_batch(batch)?;

        // Log audit events for each mutation in the batch
        for mutation in &batch.mutations {
            let op = match mutation {
                Mutation::Insert { .. } => MutationOp::Insert,
                Mutation::Update { .. } => MutationOp::Update,
                Mutation::Upsert { .. } => MutationOp::Upsert,
                Mutation::Delete { .. } => MutationOp::Delete,
            };

            self.audit_logger.log(AuditEvent::mutation(
                context,
                mutation.entity(),
                op,
                vec![], // Batch results don't track IDs per mutation
            ));
        }

        Ok(Response::mutation_ok(request_id, result))
    }

    /// Handle a get schema request.
    fn handle_get_schema(&self, request_id: u64) -> Result<Response, Error> {
        let version = self.database.schema_version();

        let data = if version == 0 {
            // No schema applied yet
            Vec::new()
        } else {
            // Get the current schema and serialize it
            let schema = self
                .database
                .catalog()
                .current_schema()
                .map_err(|e| Error::Database(format!("failed to get schema: {}", e)))?
                .ok_or_else(|| {
                    Error::Database("schema version is non-zero but no schema found".to_string())
                })?;

            schema
                .to_bytes()
                .map_err(|e| Error::Database(format!("failed to serialize schema: {}", e)))?
        };

        Ok(Response::schema_ok(request_id, version, data))
    }

    /// Handle an apply schema request.
    fn handle_apply_schema(&self, request_id: u64, bytes: &[u8]) -> Result<Response, Error> {
        use ormdb_core::catalog::SchemaBundle;

        // Deserialize the schema from bytes
        let schema = SchemaBundle::from_bytes(bytes)
            .map_err(|e| Error::Database(format!("failed to deserialize schema: {}", e)))?;

        // Apply the schema
        self.database
            .catalog()
            .apply_schema(schema)
            .map_err(|e| Error::Database(format!("failed to apply schema: {}", e)))?;

        // Return the new version
        let version = self.database.schema_version();
        Ok(Response::schema_applied_ok(request_id, version))
    }

    /// Handle an explain request.
    fn handle_explain(
        &self,
        request_id: u64,
        query: &ormdb_proto::GraphQuery,
    ) -> Result<Response, Error> {
        if let Err(e) = self
            .database
            .refresh_statistics_if_stale(STATS_REFRESH_INTERVAL)
        {
            warn!(error = %e, "Failed to refresh statistics");
        }

        let catalog = self.database.catalog();
        let statistics = self.database.statistics();
        let cache = self.database.plan_cache();

        let service = ExplainService::new(catalog)
            .with_statistics(statistics)
            .with_cache(cache);

        let result = service
            .explain(query)
            .map_err(|e| Error::Database(format!("explain failed: {}", e)))?;

        Ok(Response::explain_ok(request_id, result))
    }

    /// Handle a get metrics request.
    fn handle_get_metrics(&self, request_id: u64) -> Result<Response, Error> {
        let result = self.collect_metrics();
        Ok(Response::metrics_ok(request_id, result))
    }

    /// Collect current server metrics.
    fn collect_metrics(&self) -> MetricsResult {
        // Get metrics from registry if available
        let (uptime_secs, query_metrics, mutations, cache) = if let Some(ref registry) = self.metrics {
            let queries_by_entity: Vec<EntityQueryCount> = registry
                .queries_by_entity()
                .into_iter()
                .map(|(entity, count)| EntityQueryCount { entity, count })
                .collect();

            (
                registry.uptime_secs(),
                QueryMetrics {
                    total_count: registry.query_count(),
                    avg_duration_us: registry.avg_query_latency_us(),
                    p50_duration_us: registry.p50_query_latency_us(),
                    p99_duration_us: registry.p99_query_latency_us(),
                    max_duration_us: registry.max_query_latency_us(),
                    by_entity: queries_by_entity,
                },
                MutationMetrics {
                    total_count: registry.mutation_count(),
                    inserts: registry.insert_count(),
                    updates: registry.update_count(),
                    deletes: registry.delete_count(),
                    upserts: registry.upsert_count(),
                    rows_affected: registry.rows_affected(),
                },
                CacheMetrics {
                    hits: registry.cache_hits(),
                    misses: registry.cache_misses(),
                    hit_rate: registry.cache_hit_rate(),
                    size: self.database.plan_cache().len() as u64,
                    capacity: 1000, // Default capacity
                    evictions: registry.cache_evictions(),
                },
            )
        } else {
            // No metrics registry, return defaults
            (
                0,
                QueryMetrics::default(),
                MutationMetrics::default(),
                CacheMetrics::default(),
            )
        };

        // Get storage metrics from statistics
        let statistics = self.database.statistics();
        let entity_counts: Vec<EntityCount> = statistics
            .snapshot()
            .into_iter()
            .map(|(entity, count)| EntityCount { entity, count })
            .collect();

        let total_entities: u64 = entity_counts.iter().map(|e| e.count).sum();

        MetricsResult::new(
            uptime_secs,
            query_metrics,
            mutations,
            cache,
            StorageMetrics {
                entity_counts,
                total_entities,
                size_bytes: None,
                active_transactions: 0,
            },
            TransportMetrics::default(),
        )
    }

    /// Handle a stream changes request (CDC/replication).
    fn handle_stream_changes(
        &self,
        request_id: u64,
        req: &StreamChangesRequest,
    ) -> Result<Response, Error> {
        let changelog = self.database.changelog();

        // Scan entries from the changelog
        let (entries, has_more) = if let Some(ref filter) = req.entity_filter {
            changelog.scan_filtered(req.from_lsn, req.batch_size as usize, Some(filter))
        } else {
            changelog.scan_batch(req.from_lsn, req.batch_size as usize)
        }
        .map_err(|e| Error::Database(format!("failed to scan changelog: {}", e)))?;

        // Calculate next LSN
        let next_lsn = entries.last().map(|e| e.lsn + 1).unwrap_or(req.from_lsn);

        let response = StreamChangesResponse::new(entries, next_lsn, has_more);
        Ok(Response::stream_changes_ok(request_id, response))
    }

    /// Handle a get replication status request.
    fn handle_replication_status(&self, request_id: u64) -> Result<Response, Error> {
        let changelog = self.database.changelog();
        let current_lsn = changelog.current_lsn();

        // For now, all servers are standalone (full replication manager comes later)
        let status = ReplicationStatus::new(ReplicationRole::Standalone, current_lsn);

        Ok(Response::replication_status_ok(request_id, status))
    }

    /// Convert an error to an error response.
    ///
    /// Security note: Internal errors are logged but sanitized messages are
    /// returned to clients to prevent information disclosure.
    fn error_response(&self, request_id: u64, error: Error) -> Response {
        let (code, message) = match &error {
            Error::Database(msg) => {
                if msg.contains("not found") {
                    (error_codes::NOT_FOUND, msg.clone())
                } else {
                    // Log internal details, return sanitized message
                    warn!(error = %msg, "database error");
                    (error_codes::INTERNAL, "database operation failed".to_string())
                }
            }
            Error::Storage(e) => {
                // Log internal details, return sanitized message
                warn!(error = %e, "storage error");
                (error_codes::INTERNAL, "storage operation failed".to_string())
            }
            Error::Protocol(e) => {
                // Protocol errors are safe to return as they describe request issues
                (error_codes::INVALID_REQUEST, e.to_string())
            }
            Error::Transport(msg) => {
                // Log internal details, return sanitized message
                warn!(error = %msg, "transport error");
                (error_codes::INTERNAL, "transport error".to_string())
            }
            Error::Config(msg) => {
                // Log internal details, return sanitized message
                warn!(error = %msg, "config error");
                (error_codes::INTERNAL, "server configuration error".to_string())
            }
            Error::Io(e) => {
                // Log internal details, return sanitized message
                warn!(error = %e, "io error");
                (error_codes::INTERNAL, "I/O operation failed".to_string())
            }
            Error::Security(e) => {
                use ormdb_core::security::SecurityError;
                match e {
                    // User-facing security errors - safe to return details
                    SecurityError::AuthenticationFailed(_) => {
                        (error_codes::AUTHENTICATION_FAILED, "authentication failed".to_string())
                    }
                    SecurityError::PermissionDenied(reason) => {
                        (error_codes::PERMISSION_DENIED, format!("permission denied: {}", reason))
                    }
                    SecurityError::CapabilityNotGranted(cap) => {
                        (error_codes::PERMISSION_DENIED, format!("capability not granted: {}", cap))
                    }
                    SecurityError::RlsViolation(_) => {
                        (error_codes::PERMISSION_DENIED, "access denied by policy".to_string())
                    }
                    SecurityError::FieldAccessDenied { field, .. } => {
                        (error_codes::PERMISSION_DENIED, format!("access denied to field: {}", field))
                    }
                    SecurityError::InvalidCapabilityFormat(cap) => {
                        (error_codes::INVALID_REQUEST, format!("invalid capability format: {}", cap))
                    }
                    SecurityError::BudgetExceeded(reason) => {
                        (error_codes::BUDGET_EXCEEDED, format!("budget exceeded: {}", reason))
                    }
                    // Internal security errors - sanitize
                    SecurityError::InvalidContext(_)
                    | SecurityError::PolicyCompilationError(_)
                    | SecurityError::AuditError(_)
                    | SecurityError::Storage(_) => {
                        warn!(error = %e, "internal security error");
                        (error_codes::INTERNAL, "security operation failed".to_string())
                    }
                }
            }
        };

        Response::error(request_id, code, message)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ormdb_core::catalog::{EntityDef, FieldDef, FieldType, ScalarType, SchemaBundle};
    use ormdb_proto::{FieldValue, GraphQuery, Mutation, MutationBatch, ResponsePayload, Status};

    fn setup_test_db() -> (tempfile::TempDir, Arc<Database>) {
        let dir = tempfile::tempdir().unwrap();
        let db = Database::open(dir.path()).unwrap();

        // Create schema
        let schema = SchemaBundle::new(1).with_entity(
            EntityDef::new("User", "id")
                .with_field(FieldDef::new("id", FieldType::Scalar(ScalarType::Uuid)))
                .with_field(FieldDef::new("name", FieldType::Scalar(ScalarType::String)))
                .with_field(FieldDef::new("age", FieldType::Scalar(ScalarType::Int32))),
        );
        db.catalog().apply_schema(schema).unwrap();

        (dir, Arc::new(db))
    }

    #[test]
    fn test_ping() {
        let (_dir, db) = setup_test_db();
        let handler = RequestHandler::new(db);

        let request = Request::ping(1);
        let response = handler.handle(&request);

        assert_eq!(response.id, 1);
        assert!(response.status.is_ok());
        assert!(matches!(response.payload, ResponsePayload::Pong));
    }

    #[test]
    fn test_get_schema() {
        let (_dir, db) = setup_test_db();
        let handler = RequestHandler::new(db);

        let request = Request::get_schema(2);
        let response = handler.handle(&request);

        assert_eq!(response.id, 2);
        assert!(response.status.is_ok());

        if let ResponsePayload::Schema { version, data } = &response.payload {
            assert_eq!(*version, 1);
            assert!(!data.is_empty());
        } else {
            panic!("Expected Schema payload");
        }
    }

    #[test]
    fn test_query_empty() {
        let (_dir, db) = setup_test_db();
        let handler = RequestHandler::new(db);

        let request = Request::query(3, 1, GraphQuery::new("User"));
        let response = handler.handle(&request);

        assert_eq!(response.id, 3);
        assert!(response.status.is_ok());

        if let ResponsePayload::Query(result) = &response.payload {
            assert_eq!(result.entities.len(), 1);
            assert!(result.entities[0].is_empty());
        } else {
            panic!("Expected Query payload");
        }
    }

    #[test]
    fn test_mutation_insert() {
        let (_dir, db) = setup_test_db();
        let handler = RequestHandler::new(db);

        let mutation = Mutation::insert(
            "User",
            vec![
                FieldValue::new("name", "Alice"),
                FieldValue::new("age", 30i32),
            ],
        );
        let request = Request::mutate(4, 1, mutation);
        let response = handler.handle(&request);

        assert_eq!(response.id, 4);
        assert!(response.status.is_ok());

        if let ResponsePayload::Mutation(result) = &response.payload {
            assert_eq!(result.affected, 1);
            assert_eq!(result.inserted_ids.len(), 1);
        } else {
            panic!("Expected Mutation payload");
        }
    }

    #[test]
    fn test_mutation_batch() {
        let (_dir, db) = setup_test_db();
        let handler = RequestHandler::new(db);

        let batch = MutationBatch::from_mutations(vec![
            Mutation::insert("User", vec![FieldValue::new("name", "User1")]),
            Mutation::insert("User", vec![FieldValue::new("name", "User2")]),
        ]);
        let request = Request::mutate_batch(5, 1, batch);
        let response = handler.handle(&request);

        assert_eq!(response.id, 5);
        assert!(response.status.is_ok());

        if let ResponsePayload::Mutation(result) = &response.payload {
            assert_eq!(result.affected, 2);
            assert_eq!(result.inserted_ids.len(), 2);
        } else {
            panic!("Expected Mutation payload");
        }
    }

    #[test]
    fn test_schema_mismatch() {
        let (_dir, db) = setup_test_db();
        let handler = RequestHandler::new(db);

        // Client has wrong schema version
        let request = Request::query(6, 99, GraphQuery::new("User"));
        let response = handler.handle(&request);

        assert_eq!(response.id, 6);
        assert!(response.status.is_error());

        if let Status::Error { code, message } = &response.status {
            assert_eq!(*code, error_codes::SCHEMA_MISMATCH);
            assert!(message.contains("mismatch"));
        } else {
            panic!("Expected error status");
        }
    }

    #[test]
    fn test_insert_and_query() {
        let (_dir, db) = setup_test_db();
        let handler = RequestHandler::new(db);

        // Insert a user
        let mutation = Mutation::insert(
            "User",
            vec![
                FieldValue::new("name", "Bob"),
                FieldValue::new("age", 25i32),
            ],
        );
        let insert_request = Request::mutate(7, 1, mutation);
        let insert_response = handler.handle(&insert_request);
        assert!(insert_response.status.is_ok());

        // Query users
        let query_request = Request::query(8, 1, GraphQuery::new("User"));
        let query_response = handler.handle(&query_request);

        assert!(query_response.status.is_ok());
        if let ResponsePayload::Query(result) = &query_response.payload {
            assert_eq!(result.entities[0].len(), 1);
        } else {
            panic!("Expected Query payload");
        }
    }

    // Security tests

    #[test]
    fn test_authentication_with_api_key() {
        use crate::auth::ApiKeyAuthenticator;

        let (_dir, db) = setup_test_db();
        let auth = ApiKeyAuthenticator::new();
        auth.register_key("valid-key", vec!["read:User".to_string(), "write:User".to_string()]);

        let handler = RequestHandler::with_authenticator(db, Arc::new(auth));

        // Query with valid credentials
        let request = Request::query(1, 1, GraphQuery::new("User"))
            .with_credentials("valid-key");
        let response = handler.handle(&request);
        assert!(response.status.is_ok());
    }

    #[test]
    fn test_authentication_failure() {
        use crate::auth::ApiKeyAuthenticator;

        let (_dir, db) = setup_test_db();
        let auth = ApiKeyAuthenticator::new();
        auth.register_key("valid-key", vec!["read:User".to_string()]);

        let handler = RequestHandler::with_authenticator(db, Arc::new(auth));

        // Query with invalid credentials
        let request = Request::query(1, 1, GraphQuery::new("User"))
            .with_credentials("invalid-key");
        let response = handler.handle(&request);

        assert!(response.status.is_error());
        if let Status::Error { code, .. } = &response.status {
            assert_eq!(*code, error_codes::AUTHENTICATION_FAILED);
        }
    }

    #[test]
    fn test_permission_denied_read() {
        use crate::auth::ApiKeyAuthenticator;

        let (_dir, db) = setup_test_db();
        let auth = ApiKeyAuthenticator::new();
        // Key has write permission but not read
        auth.register_key("write-only", vec!["write:User".to_string()]);

        let handler = RequestHandler::with_authenticator(db, Arc::new(auth));

        // Query (requires read permission)
        let request = Request::query(1, 1, GraphQuery::new("User"))
            .with_credentials("write-only");
        let response = handler.handle(&request);

        assert!(response.status.is_error());
        if let Status::Error { code, .. } = &response.status {
            assert_eq!(*code, error_codes::PERMISSION_DENIED);
        }
    }

    #[test]
    fn test_permission_denied_write() {
        use crate::auth::ApiKeyAuthenticator;

        let (_dir, db) = setup_test_db();
        let auth = ApiKeyAuthenticator::new();
        // Key has read permission but not write
        auth.register_key("read-only", vec!["read:User".to_string()]);

        let handler = RequestHandler::with_authenticator(db, Arc::new(auth));

        // Insert (requires write permission)
        let mutation = Mutation::insert("User", vec![FieldValue::new("name", "Test")]);
        let request = Request::mutate(1, 1, mutation)
            .with_credentials("read-only");
        let response = handler.handle(&request);

        assert!(response.status.is_error());
        if let Status::Error { code, .. } = &response.status {
            assert_eq!(*code, error_codes::PERMISSION_DENIED);
        }
    }

    #[test]
    fn test_admin_operation_requires_admin() {
        use crate::auth::ApiKeyAuthenticator;

        let (_dir, db) = setup_test_db();
        let auth = ApiKeyAuthenticator::new();
        auth.register_key("normal-user", vec!["read:*".to_string(), "write:*".to_string()]);
        auth.register_key("admin-user", vec!["admin".to_string()]);

        let handler = RequestHandler::with_authenticator(db, Arc::new(auth));

        // Get metrics with normal user (requires admin)
        let request = Request::get_metrics(1).with_credentials("normal-user");
        let response = handler.handle(&request);
        assert!(response.status.is_error());
        if let Status::Error { code, .. } = &response.status {
            assert_eq!(*code, error_codes::PERMISSION_DENIED);
        }

        // Get metrics with admin user
        let request = Request::get_metrics(2).with_credentials("admin-user");
        let response = handler.handle(&request);
        assert!(response.status.is_ok());
    }

    #[test]
    fn test_wildcard_capability() {
        use crate::auth::ApiKeyAuthenticator;

        let (_dir, db) = setup_test_db();
        let auth = ApiKeyAuthenticator::new();
        // Key with wildcard read access
        auth.register_key("reader", vec!["read:*".to_string()]);

        let handler = RequestHandler::with_authenticator(db, Arc::new(auth));

        // Should be able to read any entity
        let request = Request::query(1, 1, GraphQuery::new("User"))
            .with_credentials("reader");
        let response = handler.handle(&request);
        assert!(response.status.is_ok());
    }
}
