//! Storage engine implementation.

use super::{
    BTreeIndex, Changelog, ColumnarStore, FullTextConfig, FullTextIndex, GeoIndex, GeoPoint,
    HashIndex, HnswConfig, Record, RTreeConfig, StorageConfig, VectorIndex, VersionedKey,
};
use crate::error::Error;
use crate::query::decode_entity;
use sled::{Db, Tree};
use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, RwLock};
use ormdb_proto::Value;

/// Tree name for entity data.
const DATA_TREE: &str = "data";

/// Tree name for metadata (latest versions, etc.).
const META_TREE: &str = "meta";

/// Tree name for entity type index.
const TYPE_INDEX_TREE: &str = "index:entity_type";

/// Prefix for latest version pointers in meta tree.
const LATEST_PREFIX: &[u8] = b"latest:";

/// The main storage engine wrapping sled.
pub struct StorageEngine {
    /// The underlying sled database.
    db: Db,

    /// Tree for entity data (versioned records).
    data_tree: Tree,

    /// Tree for metadata.
    meta_tree: Tree,

    /// Tree for entity type index (entity_type + entity_id -> empty).
    type_index_tree: Tree,

    /// Columnar store for efficient column-oriented queries.
    columnar: ColumnarStore,

    /// Hash index for O(1) equality lookups.
    hash_index: HashIndex,

    /// B-tree index for O(log N) range lookups.
    btree_index: Option<BTreeIndex>,

    /// Tracks which columns have had their B-tree index built this process.
    btree_indexed_columns: RwLock<HashSet<(String, String)>>,

    /// Vector index for approximate nearest neighbor search (HNSW).
    vector_index: Option<VectorIndex>,

    /// Geo index for spatial queries (R-tree).
    geo_index: Option<GeoIndex>,

    /// Full-text index for text search (inverted index with BM25).
    fulltext_index: Option<FullTextIndex>,

    /// Commit-timestamp oracle: last allocated commit timestamp.
    ///
    /// Held across the commit critical section so commit timestamps are
    /// allocated monotonically and the read watermark is only advanced once a
    /// transaction's data is visible. See [`super::Transaction::commit_versioned`].
    commit_clock: Mutex<u64>,

    /// Read watermark: the highest commit timestamp whose transaction is fully
    /// committed and safe to read at. A snapshot taken at this value sees a
    /// consistent prefix; any later commit gets a strictly greater timestamp.
    watermark: AtomicU64,
}

impl StorageEngine {
    /// Open or create a storage engine with the given configuration.
    pub fn open(config: StorageConfig) -> Result<Self, Error> {
        let sled_config = config.to_sled_config();
        let db = sled_config.open()?;
        let data_tree = db.open_tree(DATA_TREE)?;
        let meta_tree = db.open_tree(META_TREE)?;
        let type_index_tree = db.open_tree(TYPE_INDEX_TREE)?;
        let columnar = ColumnarStore::open(&db)?;
        let hash_index = HashIndex::open(&db)?;

        // Open B-tree index at a path alongside the sled database
        let btree_index = if let Some(path) = config.path() {
            let btree_path = path.join("btree_index");
            match BTreeIndex::open(&btree_path) {
                Ok(idx) => Some(idx),
                Err(e) => {
                    // Log but don't fail - B-tree index is optional enhancement
                    tracing::warn!("Failed to open B-tree index: {:?}", e);
                    None
                }
            }
        } else {
            None
        };

        // Initialize vector index
        let vector_index = match VectorIndex::open(&db, HnswConfig::default()) {
            Ok(vi) => Some(vi),
            Err(e) => {
                tracing::warn!("Failed to open vector index: {:?}", e);
                None
            }
        };

        // Initialize geo index
        let geo_index = match GeoIndex::open(&db, RTreeConfig::default()) {
            Ok(gi) => Some(gi),
            Err(e) => {
                tracing::warn!("Failed to open geo index: {:?}", e);
                None
            }
        };

        // Initialize full-text index
        let fulltext_index = match FullTextIndex::open(&db, FullTextConfig::default()) {
            Ok(fti) => Some(fti),
            Err(e) => {
                tracing::warn!("Failed to open full-text index: {:?}", e);
                None
            }
        };

        Ok(Self {
            db,
            data_tree,
            meta_tree,
            type_index_tree,
            columnar,
            hash_index,
            btree_index,
            btree_indexed_columns: RwLock::new(HashSet::new()),
            vector_index,
            geo_index,
            fulltext_index,
            commit_clock: Mutex::new(0),
            watermark: AtomicU64::new(0),
        })
    }

    /// Read the current commit watermark — the highest commit timestamp safe to
    /// snapshot at. Use as `read_ts` for a snapshot-consistent graph fetch.
    pub fn read_watermark(&self) -> u64 {
        self.watermark.load(Ordering::Acquire)
    }

    /// Lock the commit-timestamp oracle for the duration of a commit. The guard
    /// holds the last allocated commit timestamp.
    pub(crate) fn lock_commit_clock(&self) -> MutexGuard<'_, u64> {
        self.commit_clock.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Publish a new read watermark after a commit's data is durably visible.
    /// Must be called while holding the commit-clock lock.
    pub(crate) fn publish_watermark(&self, ts: u64) {
        self.watermark.store(ts, Ordering::Release);
    }

    /// Advance the read watermark to at least the current time, returning the new
    /// value. Call this after applying committed writes that were stamped with
    /// wall-clock version timestamps (e.g. the `MutationExecutor` direct-write
    /// path), so a subsequent snapshot read (`execute_snapshot`) observes them.
    ///
    /// Monotonic: the watermark never goes backwards. Because the just-written
    /// records carry `version_ts <= now`, reading as-of the returned watermark
    /// includes them.
    pub fn tick_watermark(&self) -> u64 {
        let mut clock = self.commit_clock.lock().unwrap_or_else(|e| e.into_inner());
        let ts = std::cmp::max(*clock + 1, super::key::current_timestamp());
        *clock = ts;
        self.watermark.store(ts, Ordering::Release);
        ts
    }

    /// Advance the read watermark to a specific, externally-chosen timestamp
    /// (monotonically — never backwards). Used on the Raft apply path where the
    /// commit timestamp is the log index, so the watermark is *deterministic*
    /// across all nodes and cluster-wide snapshot reads agree.
    pub fn advance_watermark_to(&self, ts: u64) {
        let mut clock = self.commit_clock.lock().unwrap_or_else(|e| e.into_inner());
        if ts > *clock {
            *clock = ts;
            self.watermark.store(ts, Ordering::Release);
        }
    }

    /// Check if the database was recovered from a previous crash.
    pub fn was_recovered(&self) -> bool {
        self.db.was_recovered()
    }

    /// Put a new versioned record.
    ///
    /// This creates a new version of the entity, never overwriting existing versions.
    pub fn put(&self, key: VersionedKey, record: Record) -> Result<(), Error> {
        let key_bytes = key.encode();
        let value_bytes = record.to_bytes()?;

        // Insert the versioned record
        self.data_tree.insert(key_bytes, value_bytes)?;

        // Update the latest version pointer
        self.update_latest(&key.entity_id, key.version_ts)?;

        Ok(())
    }

    /// Get a specific version of an entity.
    pub fn get(&self, entity_id: &[u8; 16], version_ts: u64) -> Result<Option<Record>, Error> {
        let key = VersionedKey::new(*entity_id, version_ts);
        let key_bytes = key.encode();

        match self.data_tree.get(key_bytes)? {
            Some(bytes) => {
                let record = Record::from_bytes(&bytes)?;
                if record.deleted {
                    Ok(None)
                } else {
                    Ok(Some(record))
                }
            }
            None => Ok(None),
        }
    }

    /// Get the latest version of an entity.
    ///
    /// Returns the version timestamp and record if found.
    pub fn get_latest(&self, entity_id: &[u8; 16]) -> Result<Option<(u64, Record)>, Error> {
        // Get the latest version timestamp from metadata
        let latest_key = self.latest_key(entity_id);
        let version_ts = match self.meta_tree.get(&latest_key)? {
            Some(bytes) => {
                let mut ts_bytes = [0u8; 8];
                ts_bytes.copy_from_slice(&bytes);
                u64::from_be_bytes(ts_bytes)
            }
            None => return Ok(None),
        };

        // Get the record at that version
        match self.get(entity_id, version_ts)? {
            Some(record) => Ok(Some((version_ts, record))),
            None => Ok(None),
        }
    }

    /// Batch fetch latest versions for multiple entity IDs.
    ///
    /// More efficient than calling `get_latest()` N times as it batches
    /// the metadata lookups. This is useful for scan operations where
    /// many entity IDs need to be resolved.
    ///
    /// Returns a Vec parallel to the input, with Some((version, record)) for found entities.
    pub fn get_latest_batch(
        &self,
        entity_ids: &[[u8; 16]],
    ) -> Result<Vec<Option<(u64, Record)>>, Error> {
        if entity_ids.is_empty() {
            return Ok(Vec::new());
        }

        // Phase 1: Batch fetch version timestamps from meta tree
        let mut version_map: Vec<Option<u64>> = Vec::with_capacity(entity_ids.len());
        for entity_id in entity_ids {
            let latest_key = self.latest_key(entity_id);
            let version = match self.meta_tree.get(&latest_key)? {
                Some(bytes) => {
                    let mut ts_bytes = [0u8; 8];
                    ts_bytes.copy_from_slice(&bytes);
                    Some(u64::from_be_bytes(ts_bytes))
                }
                None => None,
            };
            version_map.push(version);
        }

        // Phase 2: Batch fetch records from data tree
        let mut results = Vec::with_capacity(entity_ids.len());
        for (entity_id, version) in entity_ids.iter().zip(version_map.iter()) {
            let result = match version {
                Some(version_ts) => {
                    self.get(entity_id, *version_ts)?
                        .map(|record| (*version_ts, record))
                }
                None => None,
            };
            results.push(result);
        }

        Ok(results)
    }

    /// Get the version of an entity at or before a given timestamp.
    ///
    /// This is useful for point-in-time queries.
    pub fn get_at(&self, entity_id: &[u8; 16], at_ts: u64) -> Result<Option<(u64, Record)>, Error> {
        // Scan backwards from the requested timestamp
        let max_key = VersionedKey::new(*entity_id, at_ts);
        let min_key = VersionedKey::min_for_entity(*entity_id);

        for result in self
            .data_tree
            .range(min_key.encode()..=max_key.encode())
            .rev()
        {
            let (key_bytes, value_bytes) = result?;
            let key = VersionedKey::decode(&key_bytes).ok_or(Error::InvalidKey)?;

            // Verify this is for the correct entity
            if key.entity_id != *entity_id {
                continue;
            }

            let record = Record::from_bytes(&value_bytes)?;
            if !record.deleted {
                return Ok(Some((key.version_ts, record)));
            }
        }

        Ok(None)
    }

    /// Read an entity as-of a read timestamp (snapshot semantics).
    ///
    /// Returns the newest version whose `version_ts <= read_ts`. Unlike
    /// [`Self::get_at`], if that newest version is a tombstone the entity is
    /// treated as absent (`Ok(None)`) rather than resurrecting an older live
    /// version. This is the read primitive for snapshot-consistent graph
    /// fetches (milestone M4): every sub-read of a graph fetch uses the same
    /// `read_ts`, so the assembled graph corresponds to a single commit cut.
    pub fn get_as_of(
        &self,
        entity_id: &[u8; 16],
        read_ts: u64,
    ) -> Result<Option<(u64, Record)>, Error> {
        let max_key = VersionedKey::new(*entity_id, read_ts);
        let min_key = VersionedKey::min_for_entity(*entity_id);

        if let Some(result) = self
            .data_tree
            .range(min_key.encode()..=max_key.encode())
            .next_back()
        {
            let (key_bytes, value_bytes) = result?;
            let key = VersionedKey::decode(&key_bytes).ok_or(Error::InvalidKey)?;
            if key.entity_id != *entity_id {
                return Ok(None);
            }
            let record = Record::from_bytes(&value_bytes)?;
            if record.deleted {
                return Ok(None);
            }
            return Ok(Some((key.version_ts, record)));
        }

        Ok(None)
    }

    /// Batch variant of [`Self::get_as_of`]. Returns a Vec parallel to `entity_ids`.
    pub fn get_as_of_batch(
        &self,
        entity_ids: &[[u8; 16]],
        read_ts: u64,
    ) -> Result<Vec<Option<(u64, Record)>>, Error> {
        entity_ids
            .iter()
            .map(|id| self.get_as_of(id, read_ts))
            .collect()
    }

    /// Scan all entities of a type as-of a read timestamp.
    ///
    /// Yields `(entity_id, version_ts, Record)` for every entity whose state
    /// as-of `read_ts` is a live (non-tombstone) record. Entities created after
    /// `read_ts`, or deleted at-or-before it, are skipped. This is the as-of
    /// analogue of [`Self::scan_entity_type`].
    pub fn scan_entity_type_as_of(
        &self,
        entity_type: &str,
        read_ts: u64,
    ) -> impl Iterator<Item = Result<([u8; 16], u64, Record), Error>> + '_ {
        let prefix = self.type_index_prefix(entity_type);
        let prefix_len = prefix.len();

        self.type_index_tree
            .scan_prefix(&prefix)
            .filter_map(move |result| match result {
                Ok((key, _)) => {
                    if key.len() != prefix_len + 16 {
                        return Some(Err(Error::InvalidKey));
                    }
                    let mut entity_id = [0u8; 16];
                    entity_id.copy_from_slice(&key[prefix_len..]);

                    match self.get_as_of(&entity_id, read_ts) {
                        Ok(Some((version_ts, record))) => Some(Ok((entity_id, version_ts, record))),
                        Ok(None) => None,
                        Err(e) => Some(Err(e)),
                    }
                }
                Err(e) => Some(Err(e.into())),
            })
    }

    /// Scan all versions of an entity.
    ///
    /// Returns versions in chronological order (oldest first).
    pub fn scan_versions(
        &self,
        entity_id: &[u8; 16],
    ) -> impl Iterator<Item = Result<(u64, Record), Error>> + '_ {
        let min_key = VersionedKey::min_for_entity(*entity_id);
        let max_key = VersionedKey::max_for_entity(*entity_id);
        let entity_id = *entity_id;

        self.data_tree
            .range(min_key.encode()..=max_key.encode())
            .map(move |result| {
                let (key_bytes, value_bytes) = result?;
                let key = VersionedKey::decode(&key_bytes).ok_or(Error::InvalidKey)?;

                // Verify this is for the correct entity
                if key.entity_id != entity_id {
                    return Err(Error::InvalidKey);
                }

                let record = Record::from_bytes(&value_bytes)?;
                Ok((key.version_ts, record))
            })
    }

    /// Soft delete an entity by writing a tombstone record.
    pub fn delete(&self, entity_id: &[u8; 16]) -> Result<u64, Error> {
        let key = VersionedKey::now(*entity_id);
        let record = Record::tombstone();

        self.put(key, record)?;
        Ok(key.version_ts)
    }

    // ========== Entity Type-Aware Methods ==========

    /// Put a versioned record with entity type indexing.
    ///
    /// This stores the record and also indexes it by entity type for efficient scanning.
    /// Also updates the columnar store for efficient column-oriented queries when the
    /// record data is valid entity data.
    pub fn put_typed(
        &self,
        entity_type: &str,
        key: VersionedKey,
        record: Record,
    ) -> Result<(), Error> {
        // Store the record using the standard put
        self.put(key, record.clone())?;

        // Add to entity type index
        let index_key = self.type_index_key(entity_type, &key.entity_id);
        self.type_index_tree.insert(index_key, &[])?;

        // Try to decode fields for columnar storage
        // This is best-effort - if the data isn't valid entity format, skip columnar update
        if let Ok(fields) = decode_entity(&record.data) {
            if let Ok(projection) = self.columnar.projection(entity_type) {
                let _ = projection.update_row(&key.entity_id, &fields);
            }
        }

        Ok(())
    }

    /// Soft delete an entity with type indexing.
    ///
    /// Note: We don't remove from the type index because the entity still exists
    /// as a tombstone. The scan will filter out deleted entities.
    pub fn delete_typed(&self, entity_type: &str, entity_id: &[u8; 16]) -> Result<u64, Error> {
        let _ = entity_type; // Type index entry remains (can still scan history)
        self.delete(entity_id)
    }

    /// Scan all entities of a given type.
    ///
    /// Returns an iterator over (entity_id, version_ts, Record) tuples for all
    /// non-deleted entities of the specified type.
    pub fn scan_entity_type(
        &self,
        entity_type: &str,
    ) -> impl Iterator<Item = Result<([u8; 16], u64, Record), Error>> + '_ {
        let prefix = self.type_index_prefix(entity_type);
        let prefix_len = prefix.len();

        self.type_index_tree
            .scan_prefix(&prefix)
            .filter_map(move |result| {
                match result {
                    Ok((key, _)) => {
                        // Extract entity_id from index key (after the prefix)
                        if key.len() != prefix_len + 16 {
                            return Some(Err(Error::InvalidKey));
                        }
                        let mut entity_id = [0u8; 16];
                        entity_id.copy_from_slice(&key[prefix_len..]);

                        // Get the latest version of this entity
                        match self.get_latest(&entity_id) {
                            Ok(Some((version_ts, record))) => {
                                Some(Ok((entity_id, version_ts, record)))
                            }
                            Ok(None) => None, // Deleted or doesn't exist
                            Err(e) => Some(Err(e)),
                        }
                    }
                    Err(e) => Some(Err(e.into())),
                }
            })
    }

    /// Batch scan all entities of a given type.
    ///
    /// This is more efficient than `scan_entity_type()` when you need all entities,
    /// as it uses batched version lookups. Uses more memory but fewer round trips.
    ///
    /// Returns entities in index order.
    pub fn scan_entity_type_batch(
        &self,
        entity_type: &str,
    ) -> Result<Vec<([u8; 16], u64, Record)>, Error> {
        let prefix = self.type_index_prefix(entity_type);
        let prefix_len = prefix.len();

        // Phase 1: Collect all entity IDs from type index
        let entity_ids: Vec<[u8; 16]> = self.type_index_tree
            .scan_prefix(&prefix)
            .filter_map(|result| match result {
                Ok((key, _)) => {
                    if key.len() != prefix_len + 16 {
                        return None;
                    }
                    let mut entity_id = [0u8; 16];
                    entity_id.copy_from_slice(&key[prefix_len..]);
                    Some(entity_id)
                }
                Err(_) => None,
            })
            .collect();

        // Phase 2: Batch fetch latest versions
        let batch_results = self.get_latest_batch(&entity_ids)?;

        // Phase 3: Combine results, filtering out deleted/missing
        let results: Vec<([u8; 16], u64, Record)> = entity_ids
            .into_iter()
            .zip(batch_results.into_iter())
            .filter_map(|(entity_id, result)| {
                result.map(|(version_ts, record)| (entity_id, version_ts, record))
            })
            .collect();

        Ok(results)
    }

    /// Get all entity IDs of a given type (including deleted).
    ///
    /// This is useful for getting all IDs without loading the records.
    pub fn list_entity_ids(&self, entity_type: &str) -> impl Iterator<Item = Result<[u8; 16], Error>> + '_ {
        let prefix = self.type_index_prefix(entity_type);
        let prefix_len = prefix.len();

        self.type_index_tree.scan_prefix(&prefix).map(move |result| {
            let (key, _) = result?;
            if key.len() != prefix_len + 16 {
                return Err(Error::InvalidKey);
            }
            let mut entity_id = [0u8; 16];
            entity_id.copy_from_slice(&key[prefix_len..]);
            Ok(entity_id)
        })
    }

    /// Get the index key for an entity type + entity ID.
    fn type_index_key(&self, entity_type: &str, entity_id: &[u8; 16]) -> Vec<u8> {
        let mut key = Vec::with_capacity(entity_type.len() + 1 + 16);
        key.extend_from_slice(entity_type.as_bytes());
        key.push(0); // Null separator
        key.extend_from_slice(entity_id);
        key
    }

    /// Get the prefix for scanning all entities of a type.
    fn type_index_prefix(&self, entity_type: &str) -> Vec<u8> {
        let mut prefix = Vec::with_capacity(entity_type.len() + 1);
        prefix.extend_from_slice(entity_type.as_bytes());
        prefix.push(0); // Null separator
        prefix
    }

    // ========== End Entity Type-Aware Methods ==========

    /// Flush all pending writes to disk.
    pub fn flush(&self) -> Result<(), Error> {
        self.db.flush()?;
        Ok(())
    }

    /// Get database size in bytes.
    pub fn size_on_disk(&self) -> Result<u64, Error> {
        Ok(self.db.size_on_disk()?)
    }

    /// Generate a new entity ID (UUID v4 bytes).
    pub fn generate_id() -> [u8; 16] {
        use std::sync::atomic::{AtomicU64, Ordering};
        use std::time::{SystemTime, UNIX_EPOCH};

        // Counter to ensure uniqueness even with same timestamp
        static COUNTER: AtomicU64 = AtomicU64::new(0);

        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;

        // Combine timestamp with monotonically increasing counter
        let counter = COUNTER.fetch_add(1, Ordering::SeqCst);

        let mut id = [0u8; 16];
        id[..8].copy_from_slice(&now.to_le_bytes());
        id[8..16].copy_from_slice(&counter.to_le_bytes());

        // Set UUID version 4 bits
        id[6] = (id[6] & 0x0f) | 0x40;
        id[8] = (id[8] & 0x3f) | 0x80;

        id
    }

    /// Update the latest version pointer for an entity.
    fn update_latest(&self, entity_id: &[u8; 16], version_ts: u64) -> Result<(), Error> {
        let latest_key = self.latest_key(entity_id);
        self.meta_tree
            .insert(&latest_key, &version_ts.to_be_bytes())?;
        Ok(())
    }

    /// Get the metadata key for the latest version pointer.
    fn latest_key(&self, entity_id: &[u8; 16]) -> Vec<u8> {
        let mut key = Vec::with_capacity(LATEST_PREFIX.len() + 16);
        key.extend_from_slice(LATEST_PREFIX);
        key.extend_from_slice(entity_id);
        key
    }

    /// Get access to the underlying data tree (for transactions).
    pub(crate) fn data_tree(&self) -> &Tree {
        &self.data_tree
    }

    /// Get access to the underlying meta tree (for transactions).
    pub(crate) fn meta_tree(&self) -> &Tree {
        &self.meta_tree
    }

    /// Get access to the type index tree (for transactions).
    pub(crate) fn type_index_tree(&self) -> &Tree {
        &self.type_index_tree
    }

    /// Get the underlying sled database (for opening new trees).
    pub fn db(&self) -> &Db {
        &self.db
    }

    /// Get access to the columnar store.
    pub fn columnar(&self) -> &ColumnarStore {
        &self.columnar
    }

    /// Get the hash index for O(1) equality lookups.
    pub fn hash_index(&self) -> &HashIndex {
        &self.hash_index
    }

    /// Get the B-tree index for O(log N) range lookups, if available.
    pub fn btree_index(&self) -> Option<&BTreeIndex> {
        self.btree_index.as_ref()
    }

    /// Get the vector index for approximate nearest neighbor search, if available.
    pub fn vector_index(&self) -> Option<&VectorIndex> {
        self.vector_index.as_ref()
    }

    /// Get the geo index for spatial queries, if available.
    pub fn geo_index(&self) -> Option<&GeoIndex> {
        self.geo_index.as_ref()
    }

    /// Get the full-text index for text search, if available.
    pub fn fulltext_index(&self) -> Option<&FullTextIndex> {
        self.fulltext_index.as_ref()
    }

    /// Check if a vector index exists for an entity/column.
    pub fn has_vector_index(&self, entity_type: &str, column_name: &str) -> bool {
        self.vector_index
            .as_ref()
            .and_then(|vi| vi.has_index(entity_type, column_name).ok())
            .unwrap_or(false)
    }

    /// Check if a geo index exists for an entity/column.
    pub fn has_geo_index(&self, entity_type: &str, column_name: &str) -> bool {
        self.geo_index
            .as_ref()
            .and_then(|gi| gi.has_index(entity_type, column_name).ok())
            .unwrap_or(false)
    }

    /// Check if a full-text index exists for an entity/column.
    pub fn has_fulltext_index(&self, entity_type: &str, column_name: &str) -> bool {
        self.fulltext_index
            .as_ref()
            .and_then(|fti| fti.has_index(entity_type, column_name).ok())
            .unwrap_or(false)
    }

    /// Insert a vector into the vector index.
    pub fn insert_vector(
        &self,
        entity_type: &str,
        column_name: &str,
        entity_id: [u8; 16],
        vector: &[f32],
    ) -> Result<(), Error> {
        if let Some(vi) = &self.vector_index {
            vi.insert(entity_type, column_name, entity_id, vector)?;
        }
        Ok(())
    }

    /// Search for nearest neighbors in the vector index.
    pub fn search_vector(
        &self,
        entity_type: &str,
        column_name: &str,
        query: &[f32],
        k: usize,
    ) -> Result<Vec<([u8; 16], f32)>, Error> {
        if let Some(vi) = &self.vector_index {
            vi.search(entity_type, column_name, query, k)
        } else {
            Ok(Vec::new())
        }
    }

    /// Insert a geo point into the geo index.
    pub fn insert_geo_point(
        &self,
        entity_type: &str,
        column_name: &str,
        entity_id: [u8; 16],
        point: GeoPoint,
    ) -> Result<(), Error> {
        if let Some(gi) = &self.geo_index {
            gi.insert(entity_type, column_name, entity_id, point)?;
        }
        Ok(())
    }

    /// Search for entities within a radius of a center point.
    pub fn search_geo_radius(
        &self,
        entity_type: &str,
        column_name: &str,
        center: GeoPoint,
        radius_km: f64,
    ) -> Result<Vec<([u8; 16], f64)>, Error> {
        if let Some(gi) = &self.geo_index {
            gi.within_radius(entity_type, column_name, center, radius_km)
        } else {
            Ok(Vec::new())
        }
    }

    /// Insert text into the full-text index.
    pub fn insert_fulltext(
        &self,
        entity_type: &str,
        column_name: &str,
        entity_id: [u8; 16],
        text: &str,
    ) -> Result<(), Error> {
        if let Some(fti) = &self.fulltext_index {
            fti.insert(entity_type, column_name, entity_id, text)?;
        }
        Ok(())
    }

    /// Search the full-text index.
    pub fn search_fulltext(
        &self,
        entity_type: &str,
        column_name: &str,
        query: &str,
        limit: usize,
    ) -> Result<Vec<super::SearchResult>, Error> {
        if let Some(fti) = &self.fulltext_index {
            fti.search(entity_type, column_name, query, limit)
        } else {
            Ok(Vec::new())
        }
    }

    /// Lookup entity IDs by field value, merging hash index results with changelog pending entries.
    ///
    /// This is the preferred method for querying when using async index updates, as it
    /// ensures recently-written entities are included even if they haven't been indexed yet.
    ///
    /// If no changelog is provided, this falls back to a simple hash index lookup.
    pub fn lookup_with_changelog(
        &self,
        entity_type: &str,
        field: &str,
        value: &Value,
        changelog: Option<&Changelog>,
    ) -> Result<Vec<[u8; 16]>, Error> {
        // Get committed entity IDs from hash index
        let mut ids = self.hash_index.lookup(entity_type, field, value)?;

        // Merge with pending changelog entries if available
        if let Some(cl) = changelog {
            let pending_ids = cl.pending_ids_for_value(entity_type, field, value);
            if !pending_ids.is_empty() {
                // Deduplicate: add only IDs not already in the result
                let existing: std::collections::HashSet<[u8; 16]> = ids.iter().copied().collect();
                for id in pending_ids {
                    if !existing.contains(&id) {
                        ids.push(id);
                    }
                }
            }
        }

        Ok(ids)
    }

    /// Lookup entity IDs by field value with an Arc<Changelog>.
    ///
    /// Convenience method for when the changelog is held in an Arc.
    pub fn lookup_with_changelog_arc(
        &self,
        entity_type: &str,
        field: &str,
        value: &Value,
        changelog: Option<&Arc<Changelog>>,
    ) -> Result<Vec<[u8; 16]>, Error> {
        self.lookup_with_changelog(entity_type, field, value, changelog.map(|c| c.as_ref()))
    }

    /// Return the list of B-tree indexed columns for an entity.
    pub fn btree_indexed_columns_for_entity(&self, entity_type: &str) -> Vec<String> {
        let guard = match self.btree_indexed_columns.read() {
            Ok(guard) => guard,
            Err(_) => return Vec::new(),
        };

        guard
            .iter()
            .filter(|(entity, _)| entity == entity_type)
            .map(|(_, column)| column.clone())
            .collect()
    }

    /// Update hash and B-tree indexes using encoded before/after entity data.
    ///
    /// This is intended for write paths that bypass higher-level mutation helpers.
    pub fn update_secondary_indexes_from_encoded(
        &self,
        entity_type: &str,
        entity_id: [u8; 16],
        before: Option<&[u8]>,
        after: Option<&[u8]>,
        btree_columns: &[String],
    ) -> Result<(), Error> {
        let before_fields = match before {
            Some(data) => match decode_entity(data) {
                Ok(fields) => Some(fields),
                Err(_) => return Ok(()),
            },
            None => None,
        };
        let after_fields = match after {
            Some(data) => match decode_entity(data) {
                Ok(fields) => Some(fields),
                Err(_) => return Ok(()),
            },
            None => None,
        };

        self.update_secondary_indexes_from_fields(
            entity_type,
            entity_id,
            before_fields.as_deref(),
            after_fields.as_deref(),
            btree_columns,
        )?;

        Ok(())
    }

    /// Update hash and B-tree indexes using decoded before/after fields.
    ///
    /// If `after_fields` is None, the columnar row is deleted using `before_fields`.
    pub fn update_secondary_indexes_from_fields(
        &self,
        entity_type: &str,
        entity_id: [u8; 16],
        before_fields: Option<&[(String, Value)]>,
        after_fields: Option<&[(String, Value)]>,
        btree_columns: &[String],
    ) -> Result<(), Error> {
        self.update_hash_indexes_from_fields(entity_type, entity_id, before_fields, after_fields)?;
        self.update_btree_indexes_from_fields(
            entity_type,
            entity_id,
            before_fields,
            after_fields,
            btree_columns,
        )?;

        if after_fields.is_none() {
            if let Some(fields) = before_fields {
                self.delete_columnar_row_from_fields(entity_type, entity_id, fields)?;
            }
        }

        Ok(())
    }

    /// Update columnar projection using decoded fields.
    pub fn update_columnar_row_from_fields(
        &self,
        entity_type: &str,
        entity_id: [u8; 16],
        fields: &[(String, Value)],
    ) -> Result<(), Error> {
        let projection = self.columnar.projection(entity_type)?;
        projection.update_row(&entity_id, fields)?;
        Ok(())
    }

    /// Ensure a B-tree index exists for the given entity/column, building it if missing.
    ///
    /// Returns true if the B-tree index is available (and built or already present).
    pub fn ensure_btree_index(&self, entity_type: &str, column_name: &str) -> Result<bool, Error> {
        let Some(btree) = self.btree_index.as_ref() else {
            return Ok(false);
        };

        let key = (entity_type.to_string(), column_name.to_string());
        if let Ok(guard) = self.btree_indexed_columns.read() {
            if guard.contains(&key) {
                return Ok(true);
            }
        }

        let _ = btree.drop_column_index(entity_type, column_name);

        let projection = self.columnar.projection(entity_type)?;
        let indexed = btree.build_for_column(entity_type, column_name, projection.scan_column(column_name))?;

        tracing::debug!(
            entity = %entity_type,
            field = %column_name,
            indexed,
            "Built B-tree index"
        );

        if let Ok(mut guard) = self.btree_indexed_columns.write() {
            guard.insert(key);
        }

        Ok(true)
    }

    fn update_hash_indexes_from_fields(
        &self,
        entity_type: &str,
        entity_id: [u8; 16],
        before_fields: Option<&[(String, Value)]>,
        after_fields: Option<&[(String, Value)]>,
    ) -> Result<(), Error> {
        let to_map = |fields: Option<&[(String, Value)]>| -> HashMap<String, Value> {
            fields
                .unwrap_or(&[])
                .iter()
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect()
        };

        let before_map = to_map(before_fields);
        let after_map = to_map(after_fields);

        let mut names: HashSet<String> = HashSet::new();
        names.extend(before_map.keys().cloned());
        names.extend(after_map.keys().cloned());

        let hash_index = &self.hash_index;

        for name in names {
            let before_value = before_map.get(&name);
            let after_value = after_map.get(&name);

            if before_value == after_value {
                continue;
            }

            if let Some(value) = before_value {
                if !matches!(value, Value::Null) {
                    hash_index.remove(entity_type, &name, value, entity_id)?;
                }
            }

            if let Some(value) = after_value {
                if !matches!(value, Value::Null) {
                    hash_index.insert(entity_type, &name, value, entity_id)?;
                }
            }
        }

        Ok(())
    }

    fn update_btree_indexes_from_fields(
        &self,
        entity_type: &str,
        entity_id: [u8; 16],
        before_fields: Option<&[(String, Value)]>,
        after_fields: Option<&[(String, Value)]>,
        btree_columns: &[String],
    ) -> Result<(), Error> {
        let Some(btree) = self.btree_index.as_ref() else {
            return Ok(());
        };

        if btree_columns.is_empty() {
            return Ok(());
        }

        let to_map = |fields: Option<&[(String, Value)]>| -> HashMap<String, Value> {
            fields
                .unwrap_or(&[])
                .iter()
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect()
        };

        let before_map = to_map(before_fields);
        let after_map = to_map(after_fields);

        let mut columns: HashSet<String> = HashSet::new();
        columns.extend(btree_columns.iter().cloned());

        for column in columns {
            let before_value = before_map.get(&column);
            let after_value = after_map.get(&column);

            if before_value == after_value {
                continue;
            }

            if let Some(value) = before_value {
                if !matches!(value, Value::Null) {
                    btree.remove(entity_type, &column, value, entity_id)?;
                }
            }

            if let Some(value) = after_value {
                if !matches!(value, Value::Null) {
                    btree.insert(entity_type, &column, value, entity_id)?;
                }
            }
        }

        Ok(())
    }

    fn delete_columnar_row_from_fields(
        &self,
        entity_type: &str,
        entity_id: [u8; 16],
        fields: &[(String, Value)],
    ) -> Result<(), Error> {
        let projection = self.columnar.projection(entity_type)?;
        let columns: Vec<&str> = fields.iter().map(|(name, _)| name.as_str()).collect();
        projection.delete_row(&entity_id, &columns)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct TestDb {
        engine: StorageEngine,
        _dir: tempfile::TempDir, // Keep the temp dir alive
    }

    impl std::ops::Deref for TestDb {
        type Target = StorageEngine;
        fn deref(&self) -> &Self::Target {
            &self.engine
        }
    }

    fn test_engine() -> TestDb {
        let dir = tempfile::tempdir().unwrap();
        let engine = StorageEngine::open(StorageConfig::new(dir.path())).unwrap();
        TestDb { engine, _dir: dir }
    }

    #[test]
    fn test_put_and_get() {
        let engine = test_engine();
        let entity_id = StorageEngine::generate_id();
        let record = Record::new(vec![1, 2, 3, 4, 5]);
        let key = VersionedKey::now(entity_id);

        engine.put(key, record.clone()).unwrap();

        let retrieved = engine.get(&entity_id, key.version_ts).unwrap().unwrap();
        assert_eq!(retrieved.data, record.data);
    }

    #[test]
    fn test_get_latest() {
        let engine = test_engine();
        let entity_id = StorageEngine::generate_id();

        // Insert multiple versions
        let record1 = Record::new(vec![1]);
        let key1 = VersionedKey::new(entity_id, 100);
        engine.put(key1, record1).unwrap();

        let record2 = Record::new(vec![2]);
        let key2 = VersionedKey::new(entity_id, 200);
        engine.put(key2, record2.clone()).unwrap();

        let record3 = Record::new(vec![3]);
        let key3 = VersionedKey::new(entity_id, 300);
        engine.put(key3, record3.clone()).unwrap();

        // Get latest should return version 300
        let (version, latest) = engine.get_latest(&entity_id).unwrap().unwrap();
        assert_eq!(version, 300);
        assert_eq!(latest.data, vec![3]);
    }

    #[test]
    fn test_get_at_timestamp() {
        let engine = test_engine();
        let entity_id = StorageEngine::generate_id();

        // Insert versions at timestamps 100, 200, 300
        engine
            .put(VersionedKey::new(entity_id, 100), Record::new(vec![1]))
            .unwrap();
        engine
            .put(VersionedKey::new(entity_id, 200), Record::new(vec![2]))
            .unwrap();
        engine
            .put(VersionedKey::new(entity_id, 300), Record::new(vec![3]))
            .unwrap();

        // Query at timestamp 150 should return version 100
        let (version, record) = engine.get_at(&entity_id, 150).unwrap().unwrap();
        assert_eq!(version, 100);
        assert_eq!(record.data, vec![1]);

        // Query at timestamp 250 should return version 200
        let (version, record) = engine.get_at(&entity_id, 250).unwrap().unwrap();
        assert_eq!(version, 200);
        assert_eq!(record.data, vec![2]);
    }

    #[test]
    fn test_scan_versions() {
        let engine = test_engine();
        let entity_id = StorageEngine::generate_id();

        engine
            .put(VersionedKey::new(entity_id, 100), Record::new(vec![1]))
            .unwrap();
        engine
            .put(VersionedKey::new(entity_id, 200), Record::new(vec![2]))
            .unwrap();
        engine
            .put(VersionedKey::new(entity_id, 300), Record::new(vec![3]))
            .unwrap();

        let versions: Vec<_> = engine
            .scan_versions(&entity_id)
            .collect::<Result<Vec<_>, _>>()
            .unwrap();

        assert_eq!(versions.len(), 3);
        assert_eq!(versions[0].0, 100);
        assert_eq!(versions[1].0, 200);
        assert_eq!(versions[2].0, 300);
    }

    #[test]
    fn test_soft_delete() {
        let engine = test_engine();
        let entity_id = StorageEngine::generate_id();

        // Insert a record
        let key = VersionedKey::new(entity_id, 100);
        engine.put(key, Record::new(vec![1, 2, 3])).unwrap();

        // Verify it exists
        assert!(engine.get_latest(&entity_id).unwrap().is_some());

        // Soft delete
        engine.delete(&entity_id).unwrap();

        // get_latest should return None (tombstone)
        assert!(engine.get_latest(&entity_id).unwrap().is_none());

        // But we can still get the old version directly
        let old = engine.get(&entity_id, 100).unwrap().unwrap();
        assert_eq!(old.data, vec![1, 2, 3]);
    }

    #[test]
    fn test_persistence() {
        let dir = tempfile::tempdir().unwrap();
        let config = StorageConfig::new(dir.path());

        let entity_id = StorageEngine::generate_id();
        let key = VersionedKey::new(entity_id, 12345);

        // Write data
        {
            let engine = StorageEngine::open(config.clone()).unwrap();
            engine.put(key, Record::new(vec![1, 2, 3])).unwrap();
            engine.flush().unwrap();
        }

        // Reopen and verify
        {
            let engine = StorageEngine::open(config).unwrap();
            let record = engine.get(&entity_id, 12345).unwrap().unwrap();
            assert_eq!(record.data, vec![1, 2, 3]);
        }
    }

    #[test]
    fn test_put_typed_and_scan() {
        let engine = test_engine();

        // Create entities of different types
        let user1_id = StorageEngine::generate_id();
        let user2_id = StorageEngine::generate_id();
        let post1_id = StorageEngine::generate_id();

        // Insert users
        engine
            .put_typed(
                "ScanTestUser",
                VersionedKey::new(user1_id, 100),
                Record::new(vec![1]),
            )
            .unwrap();
        engine
            .put_typed(
                "ScanTestUser",
                VersionedKey::new(user2_id, 100),
                Record::new(vec![2]),
            )
            .unwrap();

        // Insert post
        engine
            .put_typed(
                "ScanTestPost",
                VersionedKey::new(post1_id, 100),
                Record::new(vec![3]),
            )
            .unwrap();

        // Flush to ensure data is persisted
        engine.flush().unwrap();

        // Scan users - should return 2
        let users: Vec<_> = engine
            .scan_entity_type("ScanTestUser")
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        assert_eq!(users.len(), 2);

        // Scan posts - should return 1
        let posts: Vec<_> = engine
            .scan_entity_type("ScanTestPost")
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        assert_eq!(posts.len(), 1);
        assert_eq!(posts[0].0, post1_id);
        assert_eq!(posts[0].2.data, vec![3]);

        // Scan unknown type - should return 0
        let comments: Vec<_> = engine
            .scan_entity_type("ScanTestComment")
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        assert_eq!(comments.len(), 0);
    }

    #[test]
    fn test_scan_excludes_deleted() {
        let engine = test_engine();

        let id1 = StorageEngine::generate_id();
        let id2 = StorageEngine::generate_id();

        // Insert two entities
        engine
            .put_typed("DeleteTestUser", VersionedKey::new(id1, 100), Record::new(vec![1]))
            .unwrap();
        engine
            .put_typed("DeleteTestUser", VersionedKey::new(id2, 100), Record::new(vec![2]))
            .unwrap();

        // Flush to ensure data is persisted
        engine.flush().unwrap();

        // Both should be returned
        let users: Vec<_> = engine
            .scan_entity_type("DeleteTestUser")
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        assert_eq!(users.len(), 2);

        // Delete one
        engine.delete_typed("DeleteTestUser", &id1).unwrap();
        engine.flush().unwrap();

        // Now only one should be returned
        let users: Vec<_> = engine
            .scan_entity_type("DeleteTestUser")
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        assert_eq!(users.len(), 1);
        assert_eq!(users[0].0, id2);
    }

    #[test]
    fn test_list_entity_ids() {
        let engine = test_engine();

        let id1 = StorageEngine::generate_id();
        let id2 = StorageEngine::generate_id();

        engine
            .put_typed("ListTestItem", VersionedKey::new(id1, 100), Record::new(vec![1]))
            .unwrap();
        engine
            .put_typed("ListTestItem", VersionedKey::new(id2, 100), Record::new(vec![2]))
            .unwrap();

        // Flush to ensure data is persisted
        engine.flush().unwrap();

        let ids: Vec<_> = engine
            .list_entity_ids("ListTestItem")
            .collect::<Result<Vec<_>, _>>()
            .unwrap();

        assert_eq!(ids.len(), 2);
        assert!(ids.contains(&id1));
        assert!(ids.contains(&id2));
    }
}
