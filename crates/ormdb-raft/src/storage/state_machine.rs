//! Raft state machine implementation for ORMDB.

use std::io::Cursor;
use std::path::PathBuf;
use std::sync::Arc;

use anyerror::AnyError;
use openraft::storage::{RaftSnapshotBuilder, RaftStateMachine, Snapshot};
use openraft::{Entry, EntryPayload, LogId, OptionalSend, StorageError, StorageIOError};
use parking_lot::RwLock;
use sled::{Db, Tree};

use ormdb_core::storage::StorageEngine;

use crate::error::RaftError;
use crate::storage::snapshot::SnapshotBuilder;
use crate::types::{ClientRequest, ClientResponse, Membership, NodeId, SnapshotMeta, StoredMembership, TypeConfig};

/// Tree name for state machine metadata.
const SM_STATE_TREE: &str = "raft_sm_state";

/// Keys in the state tree.
const KEY_LAST_APPLIED: &[u8] = b"last_applied";
const KEY_MEMBERSHIP: &[u8] = b"membership";

/// Callback type for applying mutations.
///
/// This allows the state machine to call back into the server's mutation executor
/// without creating a circular dependency.
/// The `u64` is the commit timestamp (the Raft log index), passed so the
/// application applies the entry deterministically across all nodes.
pub type ApplyMutationFn =
    Arc<dyn Fn(&ClientRequest, u64) -> Result<ClientResponse, String> + Send + Sync>;

/// Raft state machine that applies mutations to ORMDB storage.
///
/// This implements openraft's `RaftStateMachine` trait to:
/// - Apply committed log entries (mutations) to the database
/// - Create and restore snapshots
/// - Track the last applied log ID and membership configuration
pub struct OrmdbStateMachine {
    /// The ORMDB storage engine.
    storage: Arc<StorageEngine>,
    /// The sled database for metadata.
    db: Arc<Db>,
    /// State metadata tree.
    state_tree: Tree,
    /// Last applied log ID.
    last_applied: RwLock<Option<LogId<NodeId>>>,
    /// Current membership configuration.
    membership: RwLock<StoredMembership>,
    /// Snapshot directory.
    snapshot_dir: PathBuf,
    /// Callback to apply mutations.
    apply_fn: Option<ApplyMutationFn>,
}

impl OrmdbStateMachine {
    /// Create a new state machine.
    pub fn new(
        storage: Arc<StorageEngine>,
        db: Arc<Db>,
        snapshot_dir: PathBuf,
    ) -> Result<Self, RaftError> {
        let state_tree = db.open_tree(SM_STATE_TREE)?;

        // Load persisted state
        let last_applied = Self::load_last_applied(&state_tree)?;
        let membership = Self::load_membership(&state_tree)?;

        // Ensure snapshot directory exists
        std::fs::create_dir_all(&snapshot_dir).map_err(|e| RaftError::Storage(e.to_string()))?;

        Ok(Self {
            storage,
            db,
            state_tree,
            last_applied: RwLock::new(last_applied),
            membership: RwLock::new(membership),
            snapshot_dir,
            apply_fn: None,
        })
    }

    /// Set the mutation application callback.
    ///
    /// This callback is invoked for each mutation that needs to be applied.
    /// It allows the server to inject its mutation execution logic.
    pub fn with_apply_fn(mut self, apply_fn: ApplyMutationFn) -> Self {
        self.apply_fn = Some(apply_fn);
        self
    }

    /// Load last applied log ID from state tree.
    fn load_last_applied(state_tree: &Tree) -> Result<Option<LogId<NodeId>>, RaftError> {
        match state_tree.get(KEY_LAST_APPLIED)? {
            Some(bytes) => {
                let log_id: LogId<NodeId> =
                    serde_json::from_slice(&bytes).map_err(|e| RaftError::Storage(e.to_string()))?;
                Ok(Some(log_id))
            }
            None => Ok(None),
        }
    }

    /// Load membership from state tree.
    fn load_membership(state_tree: &Tree) -> Result<StoredMembership, RaftError> {
        match state_tree.get(KEY_MEMBERSHIP)? {
            Some(bytes) => {
                serde_json::from_slice(&bytes).map_err(|e| RaftError::Storage(e.to_string()))
            }
            None => Ok(StoredMembership::new(None, Membership::new(vec![], None))),
        }
    }

    /// Persist state to disk.
    fn persist_state(&self) -> Result<(), RaftError> {
        // Persist last_applied
        if let Some(log_id) = *self.last_applied.read() {
            let bytes = serde_json::to_vec(&log_id)
                .map_err(|e| RaftError::Serialization(e.to_string()))?;
            self.state_tree.insert(KEY_LAST_APPLIED, bytes)?;
        }

        // Persist membership
        let membership = self.membership.read().clone();
        let bytes =
            serde_json::to_vec(&membership).map_err(|e| RaftError::Serialization(e.to_string()))?;
        self.state_tree.insert(KEY_MEMBERSHIP, bytes)?;

        self.state_tree.flush()?;
        Ok(())
    }

    /// Apply a client request and return the response.
    ///
    /// `commit_ts` is the Raft log index of the entry; passed to the application
    /// so writes are stamped deterministically (identically on every node).
    fn apply_request(&self, request: &ClientRequest, commit_ts: u64) -> ClientResponse {
        match request {
            ClientRequest::Noop => ClientResponse::NoopResult,
            _ => {
                if let Some(apply_fn) = &self.apply_fn {
                    match apply_fn(request, commit_ts) {
                        Ok(response) => response,
                        Err(e) => ClientResponse::Error(e),
                    }
                } else {
                    // No apply function set - this is a configuration error
                    // but we handle it gracefully
                    ClientResponse::Error(
                        "State machine not configured with mutation executor".to_string(),
                    )
                }
            }
        }
    }

    /// Get the current last applied log ID.
    pub fn last_applied(&self) -> Option<LogId<NodeId>> {
        *self.last_applied.read()
    }

    /// Get the current membership.
    pub fn membership(&self) -> StoredMembership {
        self.membership.read().clone()
    }

    /// Get the storage engine.
    pub fn storage(&self) -> &Arc<StorageEngine> {
        &self.storage
    }
}

impl RaftSnapshotBuilder<TypeConfig> for OrmdbStateMachine {
    async fn build_snapshot(&mut self) -> Result<Snapshot<TypeConfig>, StorageError<NodeId>> {
        let last_applied = *self.last_applied.read();
        let membership = self.membership.read().clone();

        let mut builder =
            SnapshotBuilder::new(self.storage.clone(), self.snapshot_dir.clone(), last_applied, membership);

        builder.build_snapshot().await
    }
}

impl RaftStateMachine<TypeConfig> for OrmdbStateMachine {
    type SnapshotBuilder = Self;

    async fn applied_state(
        &mut self,
    ) -> Result<(Option<LogId<NodeId>>, StoredMembership), StorageError<NodeId>> {
        Ok((*self.last_applied.read(), self.membership.read().clone()))
    }

    async fn apply<I>(&mut self, entries: I) -> Result<Vec<ClientResponse>, StorageError<NodeId>>
    where
        I: IntoIterator<Item = Entry<TypeConfig>> + OptionalSend,
        I::IntoIter: OptionalSend,
    {
        let mut responses = Vec::new();

        for entry in entries {
            // Update last applied
            *self.last_applied.write() = Some(entry.log_id);

            let response = match entry.payload {
                EntryPayload::Blank => ClientResponse::NoopResult,
                EntryPayload::Normal(request) => {
                    self.apply_request(&request, entry.log_id.index)
                }
                EntryPayload::Membership(membership) => {
                    *self.membership.write() = StoredMembership::new(Some(entry.log_id), membership);
                    ClientResponse::NoopResult
                }
            };

            responses.push(response);
        }

        // Persist state after batch
        self.persist_state()
            .map_err(|e| StorageIOError::write_state_machine(AnyError::new(&e)))?;

        Ok(responses)
    }

    async fn get_snapshot_builder(&mut self) -> Self::SnapshotBuilder {
        // Return self as we implement RaftSnapshotBuilder
        Self {
            storage: self.storage.clone(),
            db: self.db.clone(),
            state_tree: self.state_tree.clone(),
            last_applied: RwLock::new(*self.last_applied.read()),
            membership: RwLock::new(self.membership.read().clone()),
            snapshot_dir: self.snapshot_dir.clone(),
            apply_fn: self.apply_fn.clone(),
        }
    }

    async fn begin_receiving_snapshot(
        &mut self,
    ) -> Result<Box<Cursor<Vec<u8>>>, StorageError<NodeId>> {
        Ok(Box::new(Cursor::new(Vec::new())))
    }

    async fn install_snapshot(
        &mut self,
        meta: &SnapshotMeta,
        snapshot: Box<Cursor<Vec<u8>>>,
    ) -> Result<(), StorageError<NodeId>> {
        // Update state from snapshot metadata
        *self.last_applied.write() = meta.last_log_id;
        *self.membership.write() = meta.last_membership.clone();

        // Parse and apply snapshot data
        let data = snapshot.into_inner();
        if !data.is_empty() {
            // Deserialize and restore state
            // For now, we log a warning - full implementation would restore sled trees
            tracing::info!(
                "Installing snapshot with {} bytes, last_log_id: {:?}",
                data.len(),
                meta.last_log_id
            );
        }

        // Persist the new state
        self.persist_state()
            .map_err(|e| StorageIOError::write_state_machine(AnyError::new(&e)))?;

        Ok(())
    }

    async fn get_current_snapshot(
        &mut self,
    ) -> Result<Option<Snapshot<TypeConfig>>, StorageError<NodeId>> {
        // Check if a snapshot file exists
        let snapshot_path = self.snapshot_dir.join("current.snap");
        if !snapshot_path.exists() {
            return Ok(None);
        }

        // Load snapshot metadata and data
        let meta_path = self.snapshot_dir.join("current.meta");
        if !meta_path.exists() {
            return Ok(None);
        }

        let meta_bytes = std::fs::read(&meta_path)
            .map_err(|e| StorageIOError::read_snapshot(None, AnyError::new(&e)))?;
        let meta: SnapshotMeta = serde_json::from_slice(&meta_bytes)
            .map_err(|e| StorageIOError::read_snapshot(None, AnyError::new(&e)))?;

        let data = std::fs::read(&snapshot_path)
            .map_err(|e| StorageIOError::read_snapshot(None, AnyError::new(&e)))?;

        Ok(Some(Snapshot {
            meta,
            snapshot: Box::new(Cursor::new(data)),
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use openraft::EntryPayload;
    use ormdb_core::StorageConfig;

    fn create_test_storage() -> (Arc<StorageEngine>, Arc<Db>) {
        let config = StorageConfig::temporary();
        let storage = Arc::new(StorageEngine::open(config).unwrap());
        let db = Arc::new(storage.db().clone());
        (storage, db)
    }

    fn create_test_entry(index: u64, term: u64, request: ClientRequest) -> Entry<TypeConfig> {
        Entry {
            log_id: LogId::new(openraft::CommittedLeaderId::new(term, 1), index),
            payload: EntryPayload::Normal(request),
        }
    }

    #[tokio::test]
    async fn test_state_machine_apply_noop() {
        let (storage, db) = create_test_storage();
        let snapshot_dir = tempfile::tempdir().unwrap();
        let mut sm = OrmdbStateMachine::new(storage, db, snapshot_dir.path().to_path_buf()).unwrap();

        let entry = Entry {
            log_id: LogId::new(openraft::CommittedLeaderId::new(1, 1), 1),
            payload: EntryPayload::Blank,
        };

        let responses = sm.apply(vec![entry]).await.unwrap();
        assert_eq!(responses.len(), 1);
        assert!(matches!(responses[0], ClientResponse::NoopResult));

        // Check last applied was updated
        let (last_applied, _) = sm.applied_state().await.unwrap();
        assert_eq!(last_applied.unwrap().index, 1);
    }

    #[tokio::test]
    async fn test_state_machine_membership() {
        let (storage, db) = create_test_storage();
        let snapshot_dir = tempfile::tempdir().unwrap();
        let mut sm = OrmdbStateMachine::new(storage, db, snapshot_dir.path().to_path_buf()).unwrap();

        // Apply a membership entry
        let membership = Membership::new(vec![std::collections::BTreeSet::from([1, 2, 3])], None);
        let entry = Entry {
            log_id: LogId::new(openraft::CommittedLeaderId::new(1, 1), 1),
            payload: EntryPayload::Membership(membership.clone()),
        };

        sm.apply(vec![entry]).await.unwrap();

        // Check membership was updated
        let (_, stored_membership) = sm.applied_state().await.unwrap();
        assert_eq!(*stored_membership.membership(), membership);
    }

    #[tokio::test]
    async fn test_state_machine_with_apply_fn() {
        let (storage, db) = create_test_storage();
        let snapshot_dir = tempfile::tempdir().unwrap();
        let mut sm = OrmdbStateMachine::new(storage, db, snapshot_dir.path().to_path_buf())
            .unwrap()
            .with_apply_fn(Arc::new(|_request, _commit_ts| {
                Ok(ClientResponse::mutation_result(
                    ormdb_proto::MutationResult::affected(1),
                ))
            }));

        let entry = create_test_entry(1, 1, ClientRequest::Noop);
        let responses = sm.apply(vec![entry]).await.unwrap();

        // Noop should still return NoopResult even with apply_fn
        assert!(matches!(responses[0], ClientResponse::NoopResult));
    }

    #[tokio::test]
    async fn test_state_machine_persistence() {
        let temp_dir = tempfile::tempdir().unwrap();
        let snapshot_dir = tempfile::tempdir().unwrap();

        {
            let config = StorageConfig::new(temp_dir.path());
            let storage = Arc::new(StorageEngine::open(config).unwrap());
            let db = Arc::new(storage.db().clone());
            let mut sm =
                OrmdbStateMachine::new(storage, db, snapshot_dir.path().to_path_buf())
                    .unwrap();

            // Apply an entry
            let entry = Entry {
                log_id: LogId::new(openraft::CommittedLeaderId::new(1, 1), 5),
                payload: EntryPayload::Blank,
            };
            sm.apply(vec![entry]).await.unwrap();
        }

        // Reopen and verify
        {
            let config = StorageConfig::new(temp_dir.path());
            let storage = Arc::new(StorageEngine::open(config).unwrap());
            let db = Arc::new(storage.db().clone());
            let mut sm =
                OrmdbStateMachine::new(storage, db, snapshot_dir.path().to_path_buf()).unwrap();

            let (last_applied, _) = sm.applied_state().await.unwrap();
            assert_eq!(last_applied.unwrap().index, 5);
        }
    }
}
