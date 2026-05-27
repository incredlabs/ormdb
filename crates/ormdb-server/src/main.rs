//! ORMDB Server - Standalone database server.

use std::sync::Arc;

use clap::Parser;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

use ormdb_core::metrics::new_shared_registry;
use ormdb_core::security::{CapabilityAuthenticator, DevAuthenticator};
use ormdb_server::{
    create_transport, ApiKeyAuthenticator, Args, AuthMethod, CompactionTask, Database,
    JwtAuthenticator, RequestHandler, TokenAuthenticator,
};

#[cfg(feature = "raft")]
use ormdb_raft::RaftClusterManager;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize tracing
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "ormdb_server=info".into()),
        )
        .with(tracing_subscriber::fmt::layer())
        .init();

    tracing::info!(
        version = env!("CARGO_PKG_VERSION"),
        protocol_version = ormdb_proto::PROTOCOL_VERSION,
        "starting ORMDB server"
    );

    // Parse command-line arguments
    let args = Args::parse();
    let config = args.into_config();

    tracing::info!(
        data_path = %config.data_path.display(),
        tcp_address = ?config.tcp_address,
        ipc_address = ?config.ipc_address,
        "configuration loaded"
    );

    // Open the database
    tracing::info!("opening database");
    let database = Arc::new(Database::open(&config.data_path)?);
    let schema_version = database.schema_version();
    tracing::info!(schema_version, "database opened");

    // Initialize Raft if configured
    #[cfg(feature = "raft")]
    let raft_manager = if let Some(ref raft_config) = config.raft_config {
        tracing::info!(
            node_id = raft_config.node_id,
            listen_addr = %raft_config.raft_listen_addr,
            "initializing Raft cluster mode"
        );

        let db = Arc::new(database.storage().db().clone());
        // Apply committed Raft entries to this node's local storage.
        let apply_fn = ormdb_server::make_apply_fn(database.clone());
        let manager = RaftClusterManager::new(
            raft_config.clone(),
            database.storage_arc(),
            db,
            Some(apply_fn),
        )
        .await
        .map_err(|e| Box::new(e) as Box<dyn std::error::Error>)?;

        // Initialize cluster if this is the bootstrap node
        if config.cluster_init && !config.cluster_members.is_empty() {
            tracing::info!(
                members = ?config.cluster_members,
                "initializing Raft cluster"
            );
            manager
                .initialize_cluster(config.cluster_members.clone())
                .await
                .map_err(|e| Box::new(e) as Box<dyn std::error::Error>)?;
            tracing::info!("Raft cluster initialized successfully");
        }

        tracing::info!("Raft cluster manager initialized");
        Some(Arc::new(manager))
    } else {
        None
    };

    // Create authenticator based on configuration
    let authenticator: Arc<dyn CapabilityAuthenticator + Send + Sync> = match config.auth_method {
        AuthMethod::Dev => {
            tracing::warn!("running in DEV mode - all requests have admin access. DO NOT USE IN PRODUCTION.");
            Arc::new(DevAuthenticator)
        }
        AuthMethod::ApiKey => {
            let auth = ApiKeyAuthenticator::from_default_env();
            let key_count = auth.key_count();
            if key_count == 0 {
                tracing::warn!("API key auth enabled but no keys configured. Set ORMDB_API_KEYS environment variable.");
            } else {
                tracing::info!(key_count, "API key authentication enabled");
            }
            Arc::new(auth)
        }
        AuthMethod::Token => {
            let auth = TokenAuthenticator::from_default_env();
            let token_count = auth.token_count();
            if token_count == 0 {
                tracing::warn!("token auth enabled but no tokens configured. Set ORMDB_TOKENS environment variable.");
            } else {
                tracing::info!(token_count, "token authentication enabled");
            }
            Arc::new(auth)
        }
        AuthMethod::Jwt => {
            match JwtAuthenticator::from_env() {
                Ok(auth) => {
                    tracing::info!("JWT authentication enabled");
                    Arc::new(auth)
                }
                Err(e) => {
                    tracing::error!(error = %e, "failed to initialize JWT authenticator");
                    return Err(e.into());
                }
            }
        }
    };

    // Create request handler
    let metrics = new_shared_registry();
    #[cfg(feature = "raft")]
    let handler = if raft_manager.is_some() {
        Arc::new(RequestHandler::with_full_config(
            database.clone(),
            metrics,
            authenticator,
            raft_manager.clone(),
        ))
    } else {
        Arc::new(RequestHandler::with_metrics_and_authenticator(
            database.clone(),
            metrics,
            authenticator,
        ))
    };
    #[cfg(not(feature = "raft"))]
    let handler = Arc::new(RequestHandler::with_metrics_and_authenticator(
        database.clone(),
        metrics,
        authenticator,
    ));

    // Start background compaction if enabled
    let compaction_task = config
        .compaction_interval
        .map(|interval| CompactionTask::start(database.clone(), interval));

    // Create transport
    let transport = create_transport(&config, handler)?;

    // Set up graceful shutdown
    let (shutdown_tx, shutdown_rx) = tokio::sync::broadcast::channel(1);

    // Spawn shutdown signal handler
    let shutdown_tx_clone = shutdown_tx.clone();
    tokio::spawn(async move {
        if let Err(e) = tokio::signal::ctrl_c().await {
            tracing::error!(error = %e, "failed to listen for ctrl+c");
            return;
        }
        tracing::info!("received shutdown signal");
        let _ = shutdown_tx_clone.send(());
    });

    // Run the transport
    tracing::info!("server ready, accepting connections");
    match transport.run_until_shutdown(shutdown_rx).await {
        Ok(()) => {
            tracing::info!("server shutdown complete");
        }
        Err(e) => {
            tracing::error!(error = %e, "server error");
            return Err(e.into());
        }
    }

    if let Some(task) = compaction_task {
        task.join().await;
    }

    Ok(())
}
