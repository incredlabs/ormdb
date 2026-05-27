//! Transaction support for atomic multi-key operations.

use std::collections::{HashMap, HashSet};

use super::{Record, StorageEngine, VersionedKey};
use crate::error::Error;
use crate::query::decode_entity;
use sled::transaction::{ConflictableTransactionError, TransactionalTree};
use sled::Transactional;

/// Prefix for latest version pointers in meta tree.
const LATEST_PREFIX: &[u8] = b"latest:";

/// A pending operation in a transaction.
#[derive(Debug, Clone)]
pub enum TransactionOp {
    /// Put a versioned record.
    Put {
        /// Entity type name.
        entity_type: String,
        /// Versioned key.
        key: VersionedKey,
        /// Record data.
        record: Record,
    },
    /// Delete (soft delete via tombstone).
    Delete {
        /// Entity type name.
        entity_type: String,
        /// Entity ID to delete.
        entity_id: [u8; 16],
    },
    /// Update a record (writes new version).
    Update {
        /// Entity type name.
        entity_type: String,
        /// Entity ID to update.
        entity_id: [u8; 16],
        /// New record data.
        record: Record,
    },
}

#[derive(Debug, Clone)]
struct IndexWork {
    entity_type: String,
    before: Option<Vec<u8>>,
    after: Option<Vec<u8>>,
}

/// A transaction for atomic multi-key operations.
///
/// Operations are collected and executed atomically on commit.
/// Supports read tracking for optimistic concurrency control.
pub struct Transaction<'a> {
    engine: &'a StorageEngine,
    ops: Vec<TransactionOp>,
    /// Entities read during this transaction with their versions.
    read_set: HashMap<[u8; 16], u64>,
    /// Entities being written in this transaction.
    write_set: HashSet<[u8; 16]>,
    /// Expected versions for optimistic locking.
    expected_versions: HashMap<[u8; 16], u64>,
    /// Local cache for uncommitted writes (entity_id -> record).
    write_cache: HashMap<[u8; 16], Option<Record>>,
}

impl<'a> Transaction<'a> {
    /// Create a new transaction.
    pub(crate) fn new(engine: &'a StorageEngine) -> Self {
        Self {
            engine,
            ops: Vec::new(),
            read_set: HashMap::new(),
            write_set: HashSet::new(),
            expected_versions: HashMap::new(),
            write_cache: HashMap::new(),
        }
    }

    /// Queue a put operation (legacy, without entity type).
    pub fn put(&mut self, key: VersionedKey, record: Record) -> &mut Self {
        self.write_set.insert(key.entity_id);
        self.write_cache.insert(key.entity_id, Some(record.clone()));
        self.ops.push(TransactionOp::Put {
            entity_type: String::new(),
            key,
            record,
        });
        self
    }

    /// Queue a typed put operation.
    pub fn put_typed(
        &mut self,
        entity_type: impl Into<String>,
        key: VersionedKey,
        record: Record,
    ) -> &mut Self {
        self.write_set.insert(key.entity_id);
        self.write_cache.insert(key.entity_id, Some(record.clone()));
        self.ops.push(TransactionOp::Put {
            entity_type: entity_type.into(),
            key,
            record,
        });
        self
    }

    /// Queue an insert operation (creates new entity).
    pub fn insert(
        &mut self,
        entity_type: impl Into<String>,
        entity_id: [u8; 16],
        record: Record,
    ) -> &mut Self {
        let key = VersionedKey::now(entity_id);
        self.put_typed(entity_type, key, record)
    }

    /// Queue an update operation.
    pub fn update(
        &mut self,
        entity_type: impl Into<String>,
        entity_id: [u8; 16],
        record: Record,
    ) -> &mut Self {
        self.write_set.insert(entity_id);
        self.write_cache.insert(entity_id, Some(record.clone()));
        self.ops.push(TransactionOp::Update {
            entity_type: entity_type.into(),
            entity_id,
            record,
        });
        self
    }

    /// Queue a delete operation (soft delete via tombstone).
    pub fn delete(&mut self, entity_id: [u8; 16]) -> &mut Self {
        self.write_set.insert(entity_id);
        self.write_cache.insert(entity_id, None);
        self.ops.push(TransactionOp::Delete {
            entity_type: String::new(),
            entity_id,
        });
        self
    }

    /// Queue a typed delete operation.
    pub fn delete_typed(
        &mut self,
        entity_type: impl Into<String>,
        entity_id: [u8; 16],
    ) -> &mut Self {
        self.write_set.insert(entity_id);
        self.write_cache.insert(entity_id, None);
        self.ops.push(TransactionOp::Delete {
            entity_type: entity_type.into(),
            entity_id,
        });
        self
    }

    /// Read an entity within the transaction.
    ///
    /// This returns uncommitted writes from this transaction if present,
    /// otherwise reads from the storage engine and tracks the version
    /// for conflict detection.
    pub fn read(&mut self, entity_id: &[u8; 16]) -> Result<Option<Record>, Error> {
        // Check write cache first (uncommitted writes in this tx)
        if let Some(cached) = self.write_cache.get(entity_id) {
            return Ok(cached.clone());
        }

        // Read from storage and track version
        match self.engine.get_latest(entity_id)? {
            Some((version, record)) => {
                self.read_set.insert(*entity_id, version);
                Ok(Some(record))
            }
            None => {
                // Track that we read "nothing" at version 0
                self.read_set.insert(*entity_id, 0);
                Ok(None)
            }
        }
    }

    /// Check if an entity exists (for foreign key validation).
    ///
    /// This is a read operation that tracks versions.
    pub fn exists(&mut self, entity_id: &[u8; 16]) -> Result<bool, Error> {
        // Check write cache first
        if let Some(cached) = self.write_cache.get(entity_id) {
            return Ok(cached.is_some());
        }

        // Check storage
        match self.engine.get_latest(entity_id)? {
            Some((version, _)) => {
                self.read_set.insert(*entity_id, version);
                Ok(true)
            }
            None => {
                self.read_set.insert(*entity_id, 0);
                Ok(false)
            }
        }
    }

    /// Set expected version for optimistic locking.
    ///
    /// If the entity's version doesn't match at commit time, the transaction fails.
    pub fn expect_version(&mut self, entity_id: [u8; 16], version: u64) -> &mut Self {
        self.expected_versions.insert(entity_id, version);
        self
    }

    /// Get the current version of an entity (for optimistic locking).
    pub fn get_version(&self, entity_id: &[u8; 16]) -> Result<Option<u64>, Error> {
        match self.engine.get_latest(entity_id)? {
            Some((version, _)) => Ok(Some(version)),
            None => Ok(None),
        }
    }

    /// Check if an entity is in the write set.
    pub fn is_writing(&self, entity_id: &[u8; 16]) -> bool {
        self.write_set.contains(entity_id)
    }

    /// Get the pending operations.
    pub fn operations(&self) -> &[TransactionOp] {
        &self.ops
    }

    /// Get the number of pending operations.
    pub fn operation_count(&self) -> usize {
        self.ops.len()
    }

    /// Commit the transaction atomically.
    ///
    /// All operations succeed or none do.
    /// Checks for version conflicts before committing.
    pub fn commit(self) -> Result<(), Error> {
        // Check expected versions before committing
        self.check_version_conflicts()?;

        if self.ops.is_empty() {
            return Ok(());
        }

        let index_work = self.collect_index_work()?;

        let data_tree = self.engine.data_tree();
        let meta_tree = self.engine.meta_tree();
        let type_index_tree = self.engine.type_index_tree();

        // Execute all operations in a sled transaction
        let result: Result<(), sled::transaction::TransactionError<Error>> =
            (data_tree, meta_tree, type_index_tree).transaction(|(data_tx, meta_tx, type_tx)| {
                for op in &self.ops {
                    match op {
                        TransactionOp::Put {
                            entity_type,
                            key,
                            record,
                        } => {
                            Self::execute_put(data_tx, meta_tx, key, record)?;
                            // Add to type index if entity type is specified
                            if !entity_type.is_empty() {
                                Self::execute_type_index(type_tx, entity_type, &key.entity_id)?;
                            }
                        }
                        TransactionOp::Delete {
                            entity_type: _,
                            entity_id,
                        } => {
                            Self::execute_delete(data_tx, meta_tx, entity_id)?;
                        }
                        TransactionOp::Update {
                            entity_type,
                            entity_id,
                            record,
                        } => {
                            let key = VersionedKey::now(*entity_id);
                            Self::execute_put(data_tx, meta_tx, &key, record)?;
                            // Update type index if entity type is specified
                            if !entity_type.is_empty() {
                                Self::execute_type_index(type_tx, entity_type, entity_id)?;
                            }
                        }
                    }
                }
                Ok(())
            });

        match result {
            Ok(()) => {
                self.apply_index_updates(index_work)?;
                Ok(())
            }
            Err(sled::transaction::TransactionError::Abort(e)) => Err(e),
            Err(sled::transaction::TransactionError::Storage(e)) => Err(Error::Storage(e)),
        }
    }

    /// Commit the transaction with a monotonic **commit timestamp** stamped on
    /// every write, advancing the read watermark only once the data is visible.
    ///
    /// Unlike [`Self::commit`] (which preserves each op's build-time `version_ts`),
    /// this assigns one commit timestamp `ts` to all writes in the transaction,
    /// allocated under the engine's commit-clock lock as
    /// `max(current_timestamp(), last_committed + 1)`. The lock is held across the
    /// sled commit, and the watermark is advanced to `ts` only after the commit
    /// succeeds. A reader that captures the watermark via
    /// [`StorageEngine::read_watermark`] therefore observes a consistent prefix,
    /// and any concurrent commit lands at a strictly greater `ts` — which is what
    /// makes snapshot graph fetches sound under concurrency.
    ///
    /// Returns the allocated commit timestamp.
    pub fn commit_versioned(self) -> Result<u64, Error> {
        self.check_version_conflicts()?;

        // Allocate the commit timestamp and hold the lock across the commit so
        // timestamps are monotonic and the watermark advances only when visible.
        let mut clock = self.engine.lock_commit_clock();
        let ts = std::cmp::max(crate::storage::key::current_timestamp(), *clock + 1);

        if self.ops.is_empty() {
            *clock = ts;
            self.engine.publish_watermark(ts);
            return Ok(ts);
        }

        let index_work = self.collect_index_work()?;

        let data_tree = self.engine.data_tree();
        let meta_tree = self.engine.meta_tree();
        let type_index_tree = self.engine.type_index_tree();

        let result: Result<(), sled::transaction::TransactionError<Error>> =
            (data_tree, meta_tree, type_index_tree).transaction(|(data_tx, meta_tx, type_tx)| {
                for op in &self.ops {
                    match op {
                        TransactionOp::Put { entity_type, key, record } => {
                            let stamped = VersionedKey::new(key.entity_id, ts);
                            Self::execute_put(data_tx, meta_tx, &stamped, record)?;
                            if !entity_type.is_empty() {
                                Self::execute_type_index(type_tx, entity_type, &key.entity_id)?;
                            }
                        }
                        TransactionOp::Update { entity_type, entity_id, record } => {
                            let stamped = VersionedKey::new(*entity_id, ts);
                            Self::execute_put(data_tx, meta_tx, &stamped, record)?;
                            if !entity_type.is_empty() {
                                Self::execute_type_index(type_tx, entity_type, entity_id)?;
                            }
                        }
                        TransactionOp::Delete { entity_type: _, entity_id } => {
                            let stamped = VersionedKey::new(*entity_id, ts);
                            Self::execute_put(data_tx, meta_tx, &stamped, &Record::tombstone())?;
                        }
                    }
                }
                Ok(())
            });

        match result {
            Ok(()) => {
                self.apply_index_updates(index_work)?;
                // Publish the watermark only after the data is durably visible.
                *clock = ts;
                self.engine.publish_watermark(ts);
                Ok(ts)
            }
            Err(sled::transaction::TransactionError::Abort(e)) => Err(e),
            Err(sled::transaction::TransactionError::Storage(e)) => Err(Error::Storage(e)),
        }
    }

    /// Check for version conflicts on expected versions.
    fn check_version_conflicts(&self) -> Result<(), Error> {
        for (entity_id, expected_version) in &self.expected_versions {
            let actual_version = match self.engine.get_latest(entity_id)? {
                Some((version, _)) => version,
                None => 0,
            };

            if actual_version != *expected_version {
                return Err(Error::TransactionConflict {
                    entity_id: *entity_id,
                    expected: *expected_version,
                    actual: actual_version,
                });
            }
        }
        Ok(())
    }

    fn collect_index_work(&self) -> Result<HashMap<[u8; 16], IndexWork>, Error> {
        let mut work: HashMap<[u8; 16], IndexWork> = HashMap::new();

        for op in &self.ops {
            let (entity_type, entity_id, after) = match op {
                TransactionOp::Put {
                    entity_type,
                    key,
                    record,
                } => (entity_type.as_str(), key.entity_id, Some(record.data.clone())),
                TransactionOp::Update {
                    entity_type,
                    entity_id,
                    record,
                } => (entity_type.as_str(), *entity_id, Some(record.data.clone())),
                TransactionOp::Delete {
                    entity_type,
                    entity_id,
                } => (entity_type.as_str(), *entity_id, None),
            };

            if entity_type.is_empty() {
                continue;
            }

            if let Some(entry) = work.get_mut(&entity_id) {
                entry.entity_type = entity_type.to_string();
                entry.after = after;
                continue;
            }

            let before = self
                .engine
                .get_latest(&entity_id)?
                .map(|(_, record)| record.data);

            work.insert(
                entity_id,
                IndexWork {
                    entity_type: entity_type.to_string(),
                    before,
                    after,
                },
            );
        }

        Ok(work)
    }

    fn apply_index_updates(
        &self,
        work: HashMap<[u8; 16], IndexWork>,
    ) -> Result<(), Error> {
        for (entity_id, entry) in work {
            let mut before_fields = None;
            if let Some(data) = entry.before.as_ref() {
                match decode_entity(data) {
                    Ok(fields) => {
                        before_fields = Some(fields);
                    }
                    Err(_) => {
                        if entry.after.is_none() {
                            continue;
                        }
                    }
                }
            }

            let mut after_fields = None;
            if let Some(data) = entry.after.as_ref() {
                match decode_entity(data) {
                    Ok(fields) => {
                        after_fields = Some(fields);
                    }
                    Err(_) => {
                        continue;
                    }
                }
            }

            let btree_columns = self
                .engine
                .btree_indexed_columns_for_entity(&entry.entity_type);

            self.engine.update_secondary_indexes_from_fields(
                &entry.entity_type,
                entity_id,
                before_fields.as_deref(),
                after_fields.as_deref(),
                &btree_columns,
            )?;

            if let Some(fields) = after_fields.as_ref() {
                self.engine.update_columnar_row_from_fields(
                    &entry.entity_type,
                    entity_id,
                    fields,
                )?;
            }
        }

        Ok(())
    }

    /// Rollback the transaction (discard all pending operations).
    pub fn rollback(self) {
        // Simply drop self, discarding all pending operations
        drop(self.ops);
    }

    /// Execute a type index update within a transaction.
    fn execute_type_index(
        type_tx: &TransactionalTree,
        entity_type: &str,
        entity_id: &[u8; 16],
    ) -> Result<(), ConflictableTransactionError<Error>> {
        let mut key = Vec::with_capacity(entity_type.len() + 1 + 16);
        key.extend_from_slice(entity_type.as_bytes());
        key.push(0); // Null separator
        key.extend_from_slice(entity_id);
        type_tx.insert(key, &[])?;
        Ok(())
    }

    /// Execute a put operation within a transaction.
    fn execute_put(
        data_tx: &TransactionalTree,
        meta_tx: &TransactionalTree,
        key: &VersionedKey,
        record: &Record,
    ) -> Result<(), ConflictableTransactionError<Error>> {
        let key_bytes = key.encode();
        let value_bytes = record
            .to_bytes()
            .map_err(ConflictableTransactionError::Abort)?;

        // Insert the versioned record
        data_tx.insert(&key_bytes, value_bytes)?;

        // Update the latest version pointer
        let latest_key = Self::latest_key(&key.entity_id);
        meta_tx.insert(latest_key, &key.version_ts.to_be_bytes())?;

        Ok(())
    }

    /// Execute a delete operation within a transaction.
    fn execute_delete(
        data_tx: &TransactionalTree,
        meta_tx: &TransactionalTree,
        entity_id: &[u8; 16],
    ) -> Result<(), ConflictableTransactionError<Error>> {
        let key = VersionedKey::now(*entity_id);
        let record = Record::tombstone();

        Self::execute_put(data_tx, meta_tx, &key, &record)
    }

    /// Get the metadata key for the latest version pointer.
    fn latest_key(entity_id: &[u8; 16]) -> Vec<u8> {
        let mut key = Vec::with_capacity(LATEST_PREFIX.len() + 16);
        key.extend_from_slice(LATEST_PREFIX);
        key.extend_from_slice(entity_id);
        key
    }
}

impl StorageEngine {
    /// Begin a new transaction.
    pub fn transaction(&self) -> Transaction<'_> {
        Transaction::new(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::query::encode_entity;
    use crate::storage::StorageConfig;
    use ormdb_proto::Value;

    fn test_engine() -> StorageEngine {
        StorageEngine::open(StorageConfig::temporary()).unwrap()
    }

    #[test]
    fn test_transaction_commit() {
        let engine = test_engine();
        let id1 = StorageEngine::generate_id();
        let id2 = StorageEngine::generate_id();

        // Insert two records atomically
        let mut tx = engine.transaction();
        tx.put(VersionedKey::new(id1, 100), Record::new(vec![1]));
        tx.put(VersionedKey::new(id2, 100), Record::new(vec![2]));
        tx.commit().unwrap();

        // Both should exist
        assert!(engine.get(&id1, 100).unwrap().is_some());
        assert!(engine.get(&id2, 100).unwrap().is_some());
    }

    #[test]
    fn test_transaction_rollback() {
        let engine = test_engine();
        let id1 = StorageEngine::generate_id();

        // Start a transaction but rollback
        let mut tx = engine.transaction();
        tx.put(VersionedKey::new(id1, 100), Record::new(vec![1]));
        tx.rollback();

        // Record should not exist
        assert!(engine.get(&id1, 100).unwrap().is_none());
    }

    #[test]
    fn test_transaction_multiple_versions() {
        let engine = test_engine();
        let entity_id = StorageEngine::generate_id();

        // Insert multiple versions of the same entity atomically
        let mut tx = engine.transaction();
        tx.put(VersionedKey::new(entity_id, 100), Record::new(vec![1]));
        tx.put(VersionedKey::new(entity_id, 200), Record::new(vec![2]));
        tx.put(VersionedKey::new(entity_id, 300), Record::new(vec![3]));
        tx.commit().unwrap();

        // All versions should exist
        assert_eq!(engine.get(&entity_id, 100).unwrap().unwrap().data, vec![1]);
        assert_eq!(engine.get(&entity_id, 200).unwrap().unwrap().data, vec![2]);
        assert_eq!(engine.get(&entity_id, 300).unwrap().unwrap().data, vec![3]);

        // Latest should be version 300
        let (version, record) = engine.get_latest(&entity_id).unwrap().unwrap();
        assert_eq!(version, 300);
        assert_eq!(record.data, vec![3]);
    }

    #[test]
    fn test_transaction_delete() {
        let engine = test_engine();
        let entity_id = StorageEngine::generate_id();

        // Insert a record
        engine
            .put(VersionedKey::new(entity_id, 100), Record::new(vec![1]))
            .unwrap();

        // Delete in a transaction
        let mut tx = engine.transaction();
        tx.delete(entity_id);
        tx.commit().unwrap();

        // Latest should be None (tombstone)
        assert!(engine.get_latest(&entity_id).unwrap().is_none());
    }

    #[test]
    fn test_empty_transaction() {
        let engine = test_engine();

        // Empty transaction should succeed
        let tx = engine.transaction();
        tx.commit().unwrap();
    }

    #[test]
    fn test_transaction_read_within_tx() {
        let engine = test_engine();
        let entity_id = StorageEngine::generate_id();

        // Insert a record outside transaction
        engine
            .put(VersionedKey::new(entity_id, 100), Record::new(vec![1, 2, 3]))
            .unwrap();

        // Read within a transaction
        let mut tx = engine.transaction();
        let record = tx.read(&entity_id).unwrap().unwrap();
        assert_eq!(record.data, vec![1, 2, 3]);

        // Read set should be populated
        assert!(tx.read_set.contains_key(&entity_id));
    }

    #[test]
    fn test_transaction_read_uncommitted_write() {
        let engine = test_engine();
        let entity_id = StorageEngine::generate_id();

        // Insert within transaction
        let mut tx = engine.transaction();
        tx.insert("TestEntity", entity_id, Record::new(vec![42]));

        // Read should return the uncommitted write
        let record = tx.read(&entity_id).unwrap().unwrap();
        assert_eq!(record.data, vec![42]);
    }

    #[test]
    fn test_transaction_exists() {
        let engine = test_engine();
        let id1 = StorageEngine::generate_id();
        let id2 = StorageEngine::generate_id();

        // Insert one record
        engine
            .put(VersionedKey::new(id1, 100), Record::new(vec![1]))
            .unwrap();

        let mut tx = engine.transaction();

        // Existing record should return true
        assert!(tx.exists(&id1).unwrap());

        // Non-existing record should return false
        assert!(!tx.exists(&id2).unwrap());

        // Insert id2 in transaction
        tx.insert("TestEntity", id2, Record::new(vec![2]));

        // Now id2 should exist (uncommitted)
        assert!(tx.exists(&id2).unwrap());
    }

    #[test]
    fn test_transaction_version_conflict() {
        let engine = test_engine();
        let entity_id = StorageEngine::generate_id();

        // Insert a record at version 100
        engine
            .put(VersionedKey::new(entity_id, 100), Record::new(vec![1]))
            .unwrap();

        // Start a transaction expecting version 50 (wrong)
        let mut tx = engine.transaction();
        tx.expect_version(entity_id, 50);
        tx.put(VersionedKey::new(entity_id, 200), Record::new(vec![2]));

        // Commit should fail with TransactionConflict
        let result = tx.commit();
        assert!(result.is_err());
        if let Err(Error::TransactionConflict {
            expected, actual, ..
        }) = result
        {
            assert_eq!(expected, 50);
            assert_eq!(actual, 100);
        } else {
            panic!("Expected TransactionConflict error");
        }
    }

    #[test]
    fn test_transaction_version_match() {
        let engine = test_engine();
        let entity_id = StorageEngine::generate_id();

        // Insert a record at version 100
        engine
            .put(VersionedKey::new(entity_id, 100), Record::new(vec![1]))
            .unwrap();

        // Start a transaction expecting correct version
        let mut tx = engine.transaction();
        tx.expect_version(entity_id, 100);
        tx.put(VersionedKey::new(entity_id, 200), Record::new(vec![2]));

        // Commit should succeed
        tx.commit().unwrap();

        // Verify the update
        let (version, record) = engine.get_latest(&entity_id).unwrap().unwrap();
        assert_eq!(version, 200);
        assert_eq!(record.data, vec![2]);
    }

    #[test]
    fn test_transaction_typed_put() {
        let engine = test_engine();
        let entity_id = StorageEngine::generate_id();

        // Insert with entity type
        let mut tx = engine.transaction();
        tx.put_typed(
            "TxTestUser",
            VersionedKey::new(entity_id, 100),
            Record::new(vec![1, 2, 3]),
        );
        tx.commit().unwrap();

        // Should be indexed by type
        let users: Vec<_> = engine
            .scan_entity_type("TxTestUser")
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        assert_eq!(users.len(), 1);
        assert_eq!(users[0].0, entity_id);
    }

    #[test]
    fn test_transaction_update() {
        let engine = test_engine();
        let entity_id = StorageEngine::generate_id();

        // Insert initial record
        engine
            .put_typed("UpdateTestUser", VersionedKey::new(entity_id, 100), Record::new(vec![1]))
            .unwrap();

        // Update in transaction
        let mut tx = engine.transaction();
        tx.update("UpdateTestUser", entity_id, Record::new(vec![2, 3, 4]));
        tx.commit().unwrap();

        // Verify update
        let (_, record) = engine.get_latest(&entity_id).unwrap().unwrap();
        assert_eq!(record.data, vec![2, 3, 4]);
    }

    #[test]
    fn test_transaction_delete_read_cache() {
        let engine = test_engine();
        let entity_id = StorageEngine::generate_id();

        // Insert a record
        engine
            .put(VersionedKey::new(entity_id, 100), Record::new(vec![1]))
            .unwrap();

        // Delete in transaction and verify read returns None
        let mut tx = engine.transaction();
        tx.delete(entity_id);

        // Read should return None for deleted entity
        assert!(tx.read(&entity_id).unwrap().is_none());
        assert!(!tx.exists(&entity_id).unwrap());
    }

    #[test]
    fn test_transaction_updates_indexes_and_columnar() {
        let dir = tempfile::tempdir().unwrap();
        let engine = StorageEngine::open(StorageConfig::new(dir.path())).unwrap();

        let btree_ready = engine.ensure_btree_index("User", "age").unwrap();
        assert!(btree_ready);

        let entity_id = StorageEngine::generate_id();

        let initial_fields = vec![
            ("id".to_string(), Value::Uuid(entity_id)),
            ("name".to_string(), Value::String("Alice".to_string())),
            ("age".to_string(), Value::Int32(30)),
        ];
        let initial_data = encode_entity(&initial_fields).unwrap();

        let mut tx = engine.transaction();
        tx.insert("User", entity_id, Record::new(initial_data));
        tx.commit().unwrap();

        let name_ids = engine
            .hash_index()
            .lookup("User", "name", &Value::String("Alice".to_string()))
            .unwrap();
        assert_eq!(name_ids, vec![entity_id]);

        let age_ids = engine
            .btree_index()
            .unwrap()
            .scan_equal("User", "age", &Value::Int32(30))
            .unwrap();
        assert_eq!(age_ids, vec![entity_id]);

        let projection = engine.columnar().projection("User").unwrap();
        assert_eq!(
            projection.get_column(&entity_id, "name").unwrap(),
            Some(Value::String("Alice".to_string()))
        );
        assert_eq!(
            projection.get_column(&entity_id, "age").unwrap(),
            Some(Value::Int32(30))
        );

        let updated_fields = vec![
            ("id".to_string(), Value::Uuid(entity_id)),
            ("name".to_string(), Value::String("Bob".to_string())),
            ("age".to_string(), Value::Int32(31)),
        ];
        let updated_data = encode_entity(&updated_fields).unwrap();

        let mut tx = engine.transaction();
        tx.update("User", entity_id, Record::new(updated_data));
        tx.commit().unwrap();

        let old_name_ids = engine
            .hash_index()
            .lookup("User", "name", &Value::String("Alice".to_string()))
            .unwrap();
        assert!(old_name_ids.is_empty());
        let new_name_ids = engine
            .hash_index()
            .lookup("User", "name", &Value::String("Bob".to_string()))
            .unwrap();
        assert_eq!(new_name_ids, vec![entity_id]);

        let old_age_ids = engine
            .btree_index()
            .unwrap()
            .scan_equal("User", "age", &Value::Int32(30))
            .unwrap();
        assert!(old_age_ids.is_empty());
        let new_age_ids = engine
            .btree_index()
            .unwrap()
            .scan_equal("User", "age", &Value::Int32(31))
            .unwrap();
        assert_eq!(new_age_ids, vec![entity_id]);

        assert_eq!(
            projection.get_column(&entity_id, "name").unwrap(),
            Some(Value::String("Bob".to_string()))
        );
        assert_eq!(
            projection.get_column(&entity_id, "age").unwrap(),
            Some(Value::Int32(31))
        );

        let mut tx = engine.transaction();
        tx.delete_typed("User", entity_id);
        tx.commit().unwrap();

        let deleted_name_ids = engine
            .hash_index()
            .lookup("User", "name", &Value::String("Bob".to_string()))
            .unwrap();
        assert!(deleted_name_ids.is_empty());
        let deleted_age_ids = engine
            .btree_index()
            .unwrap()
            .scan_equal("User", "age", &Value::Int32(31))
            .unwrap();
        assert!(deleted_age_ids.is_empty());
        assert_eq!(projection.get_column(&entity_id, "name").unwrap(), None);
        assert_eq!(projection.get_column(&entity_id, "age").unwrap(), None);
    }
}
