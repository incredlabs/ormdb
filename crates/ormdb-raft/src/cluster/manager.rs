//! Raft cluster manager.

use std::collections::BTreeMap;
use std::sync::Arc;

use openraft::{BasicNode, Raft, RaftMetrics};
use tokio::sync::RwLock;

use ormdb_core::storage::StorageEngine;

use crate::config::RaftConfig;
use crate::error::RaftError;
use crate::network::factory::NngNetworkFactory;
use crate::network::server::{spawn_transport, spawn_transport_with_tls};
use crate::storage::log_storage::SledRaftLogStorage;
use crate::storage::state_machine::{ApplyMutationFn, OrmdbStateMachine};
use crate::types::{ClientRequest, ClientResponse, NodeId, OrmdbRaft, TypeConfig};

/// Type alias for shared cluster manager.
pub type SharedRaftClusterManager = Arc<RaftClusterManager>;

/// A write submitted to the async writer task, carrying a synchronous reply channel.
///
/// This lets a synchronous caller (e.g. the request handler, which runs on a
/// blocking thread) submit a write to async Raft without itself touching the
/// async runtime: it hands the request to the writer task and blocks on a std
/// channel for the committed result.
struct WriteJob {
    request: ClientRequest,
    reply: std::sync::mpsc::Sender<Result<ClientResponse, RaftError>>,
}

/// Manages the Raft cluster for ORMDB.
///
/// This is the main entry point for interacting with the Raft subsystem.
/// It handles:
/// - Raft instance lifecycle
/// - Cluster initialization and membership
/// - Write request routing
/// - Leader tracking
pub struct RaftClusterManager {
    /// This node's ID.
    node_id: NodeId,
    /// The Raft instance.
    raft: Arc<OrmdbRaft>,
    /// Configuration.
    config: RaftConfig,
    /// Cached leader information.
    cached_leader: RwLock<Option<(NodeId, String)>>,
    /// Sender to the async writer task (for synchronous callers).
    write_tx: tokio::sync::mpsc::UnboundedSender<WriteJob>,
    /// Transport shutdown handle.
    _transport_shutdown: tokio::sync::oneshot::Sender<()>,
}

impl RaftClusterManager {
    /// Create a new cluster manager.
    ///
    /// This initializes the Raft instance with the provided storage and configuration.
    pub async fn new(
        config: RaftConfig,
        storage: Arc<StorageEngine>,
        db: Arc<sled::Db>,
        apply_fn: Option<ApplyMutationFn>,
    ) -> Result<Self, RaftError> {
        // Ensure data directory exists
        std::fs::create_dir_all(&config.data_dir).map_err(|e| {
            RaftError::Initialization(format!("Failed to create data dir: {}", e))
        })?;

        // Create log storage
        let log_storage = SledRaftLogStorage::open(db.clone())?;

        // Create state machine
        let snapshot_dir = config.data_dir.join("snapshots");
        let state_machine = if let Some(apply_fn) = apply_fn {
            OrmdbStateMachine::new(storage, db, snapshot_dir)?.with_apply_fn(apply_fn)
        } else {
            OrmdbStateMachine::new(storage, db, snapshot_dir)?
        };

        // Create Raft config
        let raft_config = Arc::new(
            openraft::Config {
                cluster_name: "ormdb-cluster".to_string(),
                heartbeat_interval: config.heartbeat_interval_ms,
                election_timeout_min: config.election_timeout_min_ms,
                election_timeout_max: config.election_timeout_max_ms,
                max_payload_entries: config.max_entries_per_append,
                snapshot_policy: openraft::SnapshotPolicy::LogsSinceLast(
                    config.snapshot_threshold,
                ),
                ..Default::default()
            }
            .validate()
            .map_err(|e| RaftError::Initialization(e.to_string()))?,
        );

        // Create network factory with TLS if configured
        let network = if config.tls.enabled {
            NngNetworkFactory::with_tls(config.node_id, config.tls.clone())
        } else {
            NngNetworkFactory::new(config.node_id)
        };

        // Create Raft instance
        let raft = Raft::new(
            config.node_id,
            raft_config,
            network,
            log_storage,
            state_machine,
        )
        .await
        .map_err(|e| RaftError::Initialization(e.to_string()))?;

        let raft = Arc::new(raft);

        // Spawn an async writer task. Synchronous callers submit writes via
        // `write_blocking`, which forwards them here and blocks on a std channel.
        let (write_tx, mut write_rx) = tokio::sync::mpsc::unbounded_channel::<WriteJob>();
        {
            let raft = raft.clone();
            tokio::spawn(async move {
                while let Some(job) = write_rx.recv().await {
                    let result = raft
                        .client_write(job.request)
                        .await
                        .map(|r| r.data)
                        .map_err(|e| RaftError::Write(e.to_string()));
                    let _ = job.reply.send(result);
                }
            });
        }

        // Start transport server (with TLS if configured)
        let (_, shutdown_tx) = if config.tls.enabled {
            spawn_transport_with_tls(
                config.node_id,
                &config.raft_listen_addr,
                raft.clone(),
                config.tls.clone(),
            )
        } else {
            spawn_transport(config.node_id, &config.raft_listen_addr, raft.clone())
        };

        Ok(Self {
            node_id: config.node_id,
            raft,
            config,
            cached_leader: RwLock::new(None),
            write_tx,
            _transport_shutdown: shutdown_tx,
        })
    }

    /// Initialize the cluster with initial members.
    ///
    /// This should be called on one node (typically the first) to bootstrap
    /// the cluster. Other nodes should join using `add_learner` and
    /// `change_membership`.
    pub async fn initialize_cluster(
        &self,
        members: Vec<(NodeId, String)>,
    ) -> Result<(), RaftError> {
        let nodes: BTreeMap<NodeId, BasicNode> = members
            .into_iter()
            .map(|(id, addr)| (id, BasicNode { addr }))
            .collect();

        self.raft
            .initialize(nodes)
            .await
            .map_err(|e| RaftError::Initialization(e.to_string()))?;

        tracing::info!("Cluster initialized on node {}", self.node_id);
        Ok(())
    }

    /// Submit a client request (mutation) to the cluster.
    ///
    /// This will only succeed if this node is the leader. Otherwise,
    /// returns an error with the current leader information.
    pub async fn write(&self, request: ClientRequest) -> Result<ClientResponse, RaftError> {
        let result = self
            .raft
            .client_write(request)
            .await
            .map_err(|e| RaftError::Write(e.to_string()))?;

        Ok(result.data)
    }

    /// Submit a client request and block until it is committed and applied.
    ///
    /// Safe to call from a synchronous context (the request handler runs on a
    /// blocking thread): the request is forwarded to the async writer task and the
    /// caller blocks on a std channel for the reply, so this never touches the
    /// async runtime from the calling thread. Only succeeds on the leader.
    pub fn write_blocking(&self, request: ClientRequest) -> Result<ClientResponse, RaftError> {
        let (reply_tx, reply_rx) = std::sync::mpsc::channel();
        self.write_tx
            .send(WriteJob {
                request,
                reply: reply_tx,
            })
            .map_err(|_| RaftError::Internal("raft writer task is not running".to_string()))?;
        reply_rx
            .recv()
            .map_err(|_| RaftError::Internal("raft writer task dropped the reply".to_string()))?
    }

    /// Check if this node is the leader.
    pub fn is_leader(&self) -> bool {
        let metrics = self.raft.metrics().borrow().clone();
        metrics.current_leader == Some(self.node_id)
    }

    /// Get the current leader.
    ///
    /// Returns the leader's node ID and address if known.
    pub async fn get_leader(&self) -> Option<(NodeId, String)> {
        // First check cache
        if let Some(cached) = self.cached_leader.read().await.clone() {
            return Some(cached);
        }

        // Get from metrics
        let metrics = self.raft.metrics().borrow().clone();

        if let Some(leader_id) = metrics.current_leader {
            // Get leader address from membership
            let membership = metrics.membership_config.membership();
            if let Some(node) = membership.get_node(&leader_id) {
                let leader_info = (leader_id, node.addr.clone());

                // Update cache
                *self.cached_leader.write().await = Some(leader_info.clone());

                return Some(leader_info);
            }
        }

        None
    }

    /// Add a new node to the cluster as a learner.
    ///
    /// Learners receive log replication but don't vote.
    /// Use `change_membership` to promote to voter.
    pub async fn add_learner(&self, node_id: NodeId, addr: String) -> Result<(), RaftError> {
        self.raft
            .add_learner(node_id, BasicNode { addr }, true)
            .await
            .map_err(|e| RaftError::MembershipChange(e.to_string()))?;

        tracing::info!("Added learner node {} to cluster", node_id);
        Ok(())
    }

    /// Change cluster membership.
    ///
    /// This promotes learners to voters or removes nodes.
    pub async fn change_membership(&self, members: Vec<NodeId>) -> Result<(), RaftError> {
        use std::collections::BTreeSet;
        let member_set: BTreeSet<NodeId> = members.into_iter().collect();

        self.raft
            .change_membership(member_set, false)
            .await
            .map_err(|e| RaftError::MembershipChange(e.to_string()))?;

        tracing::info!("Membership change completed");
        Ok(())
    }

    /// Get Raft metrics for monitoring.
    pub fn metrics(&self) -> RaftMetrics<NodeId, BasicNode> {
        self.raft.metrics().borrow().clone()
    }

    /// Get this node's ID.
    pub fn node_id(&self) -> NodeId {
        self.node_id
    }

    /// Get the Raft instance.
    pub fn raft(&self) -> &Arc<OrmdbRaft> {
        &self.raft
    }

    /// Get the configuration.
    pub fn config(&self) -> &RaftConfig {
        &self.config
    }

    /// Shutdown the Raft node gracefully.
    pub async fn shutdown(&self) -> Result<(), RaftError> {
        self.raft
            .shutdown()
            .await
            .map_err(|e| RaftError::Internal(format!("Shutdown failed: {}", e)))?;

        tracing::info!("Raft node {} shut down", self.node_id);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{RaftConfig, RaftTlsConfig};
    use ormdb_core::StorageConfig;
    use std::path::PathBuf;

    fn create_test_config(node_id: u64, port: u16) -> RaftConfig {
        RaftConfig {
            node_id,
            raft_listen_addr: format!("127.0.0.1:{}", port),
            raft_advertise_addr: format!("127.0.0.1:{}", port),
            heartbeat_interval_ms: 100,
            election_timeout_min_ms: 200,
            election_timeout_max_ms: 400,
            snapshot_threshold: 1000,
            max_entries_per_append: 100,
            data_dir: PathBuf::from("/tmp/ormdb-test"),
            tls: RaftTlsConfig::default(),
        }
    }

    // Integration tests would go here
    // They require multiple nodes and would be in a separate integration test file
}
