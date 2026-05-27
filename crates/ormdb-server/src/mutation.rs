//! Mutation executor for handling write operations.

use ormdb_core::query::{decode_entity, encode_entity};
use ormdb_core::replication::ChangeLog;
use ormdb_core::storage::{Record, StorageEngine, VersionedKey};
use ormdb_proto::replication::ChangeLogEntry;
use ormdb_proto::{FieldValue, Mutation, MutationBatch, MutationResult, Value};

use crate::database::Database;
use crate::error::Error;

/// Executes mutation operations against the database.
pub struct MutationExecutor<'a> {
    database: &'a Database,
}

impl<'a> MutationExecutor<'a> {
    /// Create a new mutation executor.
    pub fn new(database: &'a Database) -> Self {
        Self { database }
    }

    /// Execute a single mutation (without CDC logging).
    pub fn execute(&self, mutation: &Mutation) -> Result<MutationResult, Error> {
        let result = match mutation {
            Mutation::Insert { entity, data } => self.execute_insert(entity, data),
            Mutation::Update { entity, id, data } => self.execute_update(entity, id, data),
            Mutation::Delete { entity, id } => self.execute_delete(entity, id),
            Mutation::Upsert { entity, id, data } => self.execute_upsert(entity, id.as_ref(), data),
        }?;
        // Advance the read watermark so snapshot reads observe this write.
        self.database.storage().tick_watermark();
        Ok(result)
    }

    /// Execute a single mutation with CDC logging.
    ///
    /// This logs the mutation to the changelog after successful execution.
    pub fn execute_with_cdc(&self, mutation: &Mutation) -> Result<MutationResult, Error> {
        let changelog = self.database.changelog();
        let schema_version = self.database.schema_version();

        let result = match mutation {
            Mutation::Insert { entity, data } => {
                self.execute_insert_with_cdc(entity, data, changelog, schema_version)
            }
            Mutation::Update { entity, id, data } => {
                self.execute_update_with_cdc(entity, id, data, changelog, schema_version)
            }
            Mutation::Delete { entity, id } => {
                self.execute_delete_with_cdc(entity, id, changelog, schema_version)
            }
            Mutation::Upsert { entity, id, data } => {
                self.execute_upsert_with_cdc(entity, id.as_ref(), data, changelog, schema_version)
            }
        }?;
        // Advance the read watermark so snapshot reads observe this write.
        self.database.storage().tick_watermark();
        Ok(result)
    }

    /// Execute a batch of mutations atomically (without CDC logging).
    pub fn execute_batch(&self, batch: &MutationBatch) -> Result<MutationResult, Error> {
        if batch.is_empty() {
            return Ok(MutationResult::affected(0));
        }

        let mut inserted_ids = Vec::new();
        let mut affected = 0u64;

        // Execute each mutation in order
        // Note: For true atomicity, we'd use sled transactions here.
        // For now, we execute sequentially which is sufficient for most use cases.
        for mutation in &batch.mutations {
            let result = self.execute(mutation)?;
            affected += result.affected;
            inserted_ids.extend(result.inserted_ids);
        }

        if inserted_ids.is_empty() {
            Ok(MutationResult::affected(affected))
        } else {
            Ok(MutationResult::bulk_inserted(inserted_ids))
        }
    }

    /// Execute a batch of mutations with CDC logging.
    pub fn execute_batch_with_cdc(&self, batch: &MutationBatch) -> Result<MutationResult, Error> {
        if batch.is_empty() {
            return Ok(MutationResult::affected(0));
        }

        let mut inserted_ids = Vec::new();
        let mut affected = 0u64;

        for mutation in &batch.mutations {
            let result = self.execute_with_cdc(mutation)?;
            affected += result.affected;
            inserted_ids.extend(result.inserted_ids);
        }

        if inserted_ids.is_empty() {
            Ok(MutationResult::affected(affected))
        } else {
            Ok(MutationResult::bulk_inserted(inserted_ids))
        }
    }

    /// Execute an insert operation.
    fn execute_insert(&self, entity: &str, data: &[FieldValue]) -> Result<MutationResult, Error> {
        let (id, _encoded) = self.do_insert(entity, data)?;
        Ok(MutationResult::inserted(id))
    }

    /// Execute an insert operation with CDC logging.
    fn execute_insert_with_cdc(
        &self,
        entity: &str,
        data: &[FieldValue],
        changelog: &ChangeLog,
        schema_version: u64,
    ) -> Result<MutationResult, Error> {
        let (id, encoded) = self.do_insert(entity, data)?;

        // Create changelog entry
        let changed_fields: Vec<String> = data.iter().map(|fv| fv.field.clone()).collect();
        let entry = ChangeLogEntry::insert(entity, id, encoded, changed_fields, schema_version);

        // Log to changelog
        changelog
            .append(entry)
            .map_err(|e| Error::Database(format!("failed to log to changelog: {}", e)))?;

        Ok(MutationResult::inserted(id))
    }

    /// Internal insert implementation that returns both ID and encoded data.
    fn do_insert(&self, entity: &str, data: &[FieldValue]) -> Result<([u8; 16], Vec<u8>), Error> {
        // Generate a new entity ID
        let id = StorageEngine::generate_id();

        // Build field data including the ID
        let mut fields: Vec<(String, Value)> = Vec::with_capacity(data.len() + 1);
        fields.push(("id".to_string(), Value::Uuid(id)));

        for fv in data {
            // Skip if someone tried to set the ID field
            if fv.field != "id" {
                fields.push((fv.field.clone(), fv.value.clone()));
            }
        }

        // Encode the entity data
        let encoded = encode_entity(&fields)
            .map_err(|e| Error::Database(format!("failed to encode entity: {}", e)))?;

        // Store the entity
        let key = VersionedKey::now(id);
        self.database
            .storage()
            .put_typed(entity, key, Record::new(encoded.clone()))
            .map_err(|e| Error::Storage(e))?;
        self.database.statistics().increment(entity);
        self.update_secondary_indexes(entity, id, None, Some(&encoded))?;

        Ok((id, encoded))
    }

    /// Execute an update operation.
    fn execute_update(
        &self,
        entity: &str,
        id: &[u8; 16],
        data: &[FieldValue],
    ) -> Result<MutationResult, Error> {
        let _ = self.do_update(entity, id, data)?;
        Ok(MutationResult::affected(1))
    }

    /// Execute an update operation with CDC logging.
    fn execute_update_with_cdc(
        &self,
        entity: &str,
        id: &[u8; 16],
        data: &[FieldValue],
        changelog: &ChangeLog,
        schema_version: u64,
    ) -> Result<MutationResult, Error> {
        let (before_data, after_data, changed_fields) = self.do_update(entity, id, data)?;

        // Create changelog entry
        let entry = ChangeLogEntry::update(
            entity,
            *id,
            before_data,
            after_data,
            changed_fields,
            schema_version,
        );

        // Log to changelog
        changelog
            .append(entry)
            .map_err(|e| Error::Database(format!("failed to log to changelog: {}", e)))?;

        Ok(MutationResult::affected(1))
    }

    /// Internal update implementation that returns before/after data and changed fields.
    fn do_update(
        &self,
        entity: &str,
        id: &[u8; 16],
        data: &[FieldValue],
    ) -> Result<(Vec<u8>, Vec<u8>, Vec<String>), Error> {
        // Get existing entity data
        let (_version, existing) = self
            .database
            .storage()
            .get_latest(id)
            .map_err(|e| Error::Storage(e))?
            .ok_or_else(|| {
                Error::Database(format!("entity {}:{} not found", entity, hex_id(id)))
            })?;

        let before_data = existing.data.clone();

        // Decode existing fields
        let before_fields: Vec<(String, Value)> =
            decode_entity(&existing.data)
                .map_err(|e| Error::Database(format!("failed to decode entity: {}", e)))?;

        let mut fields = before_fields.clone();

        // Merge updates (replace existing fields, add new ones)
        let mut changed_fields = Vec::new();
        for fv in data {
            // Don't allow updating the ID field
            if fv.field == "id" {
                continue;
            }

            if let Some(pos) = fields.iter().position(|(name, _)| name == &fv.field) {
                if fields[pos].1 != fv.value {
                    changed_fields.push(fv.field.clone());
                }
                fields[pos].1 = fv.value.clone();
            } else {
                changed_fields.push(fv.field.clone());
                fields.push((fv.field.clone(), fv.value.clone()));
            }
        }

        // Encode and store
        let encoded = encode_entity(&fields)
            .map_err(|e| Error::Database(format!("failed to encode entity: {}", e)))?;

        let key = VersionedKey::now(*id);
        self.database
            .storage()
            .put_typed(entity, key, Record::new(encoded.clone()))
            .map_err(|e| Error::Storage(e))?;
        self.update_secondary_indexes(entity, *id, Some(&before_data), Some(&encoded))?;

        Ok((before_data, encoded, changed_fields))
    }

    /// Execute a delete operation.
    fn execute_delete(&self, entity: &str, id: &[u8; 16]) -> Result<MutationResult, Error> {
        let before_data = self.do_delete(entity, id)?;
        if before_data.is_some() {
            Ok(MutationResult::affected(1))
        } else {
            Ok(MutationResult::affected(0))
        }
    }

    /// Execute a delete operation with CDC logging.
    fn execute_delete_with_cdc(
        &self,
        entity: &str,
        id: &[u8; 16],
        changelog: &ChangeLog,
        schema_version: u64,
    ) -> Result<MutationResult, Error> {
        let before_data = self.do_delete(entity, id)?;

        if let Some(data) = before_data {
            // Create changelog entry
            let entry = ChangeLogEntry::delete(entity, *id, data, schema_version);

            // Log to changelog
            changelog
                .append(entry)
                .map_err(|e| Error::Database(format!("failed to log to changelog: {}", e)))?;

            Ok(MutationResult::affected(1))
        } else {
            Ok(MutationResult::affected(0))
        }
    }

    /// Internal delete implementation that returns the before data if entity existed.
    fn do_delete(&self, entity: &str, id: &[u8; 16]) -> Result<Option<Vec<u8>>, Error> {
        // Check if entity exists and get its data
        let existing = self
            .database
            .storage()
            .get_latest(id)
            .map_err(|e| Error::Storage(e))?;

        if existing.is_none() {
            return Ok(None);
        }

        let (_version, record) = existing.unwrap();
        let before_data = record.data.clone();

        // Soft delete (creates tombstone)
        self.database
            .storage()
            .delete_typed(entity, id)
            .map_err(|e| Error::Storage(e))?;
        self.database.statistics().decrement(entity);
        self.update_secondary_indexes(entity, *id, Some(&before_data), None)?;

        Ok(Some(before_data))
    }

    /// Execute an upsert operation.
    fn execute_upsert(
        &self,
        entity: &str,
        id: Option<&[u8; 16]>,
        data: &[FieldValue],
    ) -> Result<MutationResult, Error> {
        match id {
            Some(existing_id) => {
                // Check if entity exists
                let exists = self
                    .database
                    .storage()
                    .get_latest(existing_id)
                    .map_err(|e| Error::Storage(e))?
                    .is_some();

                if exists {
                    // Update existing
                    self.execute_update(entity, existing_id, data)
                } else {
                    // Insert with provided ID
                    self.execute_insert_with_id(entity, *existing_id, data)
                }
            }
            None => {
                // No ID provided, always insert
                self.execute_insert(entity, data)
            }
        }
    }

    /// Execute an upsert operation with CDC logging.
    fn execute_upsert_with_cdc(
        &self,
        entity: &str,
        id: Option<&[u8; 16]>,
        data: &[FieldValue],
        changelog: &ChangeLog,
        schema_version: u64,
    ) -> Result<MutationResult, Error> {
        match id {
            Some(existing_id) => {
                // Check if entity exists
                let exists = self
                    .database
                    .storage()
                    .get_latest(existing_id)
                    .map_err(|e| Error::Storage(e))?
                    .is_some();

                if exists {
                    // Update existing with CDC
                    self.execute_update_with_cdc(entity, existing_id, data, changelog, schema_version)
                } else {
                    // Insert with provided ID and CDC
                    self.execute_insert_with_id_cdc(entity, *existing_id, data, changelog, schema_version)
                }
            }
            None => {
                // No ID provided, always insert with CDC
                self.execute_insert_with_cdc(entity, data, changelog, schema_version)
            }
        }
    }

    /// Execute an insert with a specific ID.
    fn execute_insert_with_id(
        &self,
        entity: &str,
        id: [u8; 16],
        data: &[FieldValue],
    ) -> Result<MutationResult, Error> {
        let _ = self.do_insert_with_id(entity, id, data)?;
        Ok(MutationResult::inserted(id))
    }

    /// Execute an insert with a specific ID and CDC logging.
    fn execute_insert_with_id_cdc(
        &self,
        entity: &str,
        id: [u8; 16],
        data: &[FieldValue],
        changelog: &ChangeLog,
        schema_version: u64,
    ) -> Result<MutationResult, Error> {
        let encoded = self.do_insert_with_id(entity, id, data)?;

        // Create changelog entry
        let changed_fields: Vec<String> = data.iter().map(|fv| fv.field.clone()).collect();
        let entry = ChangeLogEntry::insert(entity, id, encoded, changed_fields, schema_version);

        // Log to changelog
        changelog
            .append(entry)
            .map_err(|e| Error::Database(format!("failed to log to changelog: {}", e)))?;

        Ok(MutationResult::inserted(id))
    }

    /// Internal insert with ID implementation.
    fn do_insert_with_id(
        &self,
        entity: &str,
        id: [u8; 16],
        data: &[FieldValue],
    ) -> Result<Vec<u8>, Error> {
        // Build field data including the ID
        let mut fields: Vec<(String, Value)> = Vec::with_capacity(data.len() + 1);
        fields.push(("id".to_string(), Value::Uuid(id)));

        for fv in data {
            if fv.field != "id" {
                fields.push((fv.field.clone(), fv.value.clone()));
            }
        }

        // Encode the entity data
        let encoded = encode_entity(&fields)
            .map_err(|e| Error::Database(format!("failed to encode entity: {}", e)))?;

        // Store the entity
        let key = VersionedKey::now(id);
        self.database
            .storage()
            .put_typed(entity, key, Record::new(encoded.clone()))
            .map_err(|e| Error::Storage(e))?;
        self.database.statistics().increment(entity);
        self.update_secondary_indexes(entity, id, None, Some(&encoded))?;

        Ok(encoded)
    }

    fn update_secondary_indexes(
        &self,
        entity: &str,
        entity_id: [u8; 16],
        before: Option<&[u8]>,
        after: Option<&[u8]>,
    ) -> Result<(), Error> {
        let before_fields = if let Some(data) = before {
            Some(decode_entity(data).map_err(|e| {
                Error::Database(format!("failed to decode entity: {}", e))
            })?)
        } else {
            None
        };

        let after_fields = if let Some(data) = after {
            Some(decode_entity(data).map_err(|e| {
                Error::Database(format!("failed to decode entity: {}", e))
            })?)
        } else {
            None
        };

        self.update_hash_indexes(entity, entity_id, &before_fields, &after_fields)?;
        self.update_btree_indexes(entity, entity_id, &before_fields, &after_fields)?;

        if after.is_none() {
            self.delete_columnar_row(entity, entity_id)?;
        }

        Ok(())
    }

    fn update_hash_indexes(
        &self,
        entity: &str,
        entity_id: [u8; 16],
        before_fields: &Option<Vec<(String, Value)>>,
        after_fields: &Option<Vec<(String, Value)>>,
    ) -> Result<(), Error> {
        use std::collections::{HashMap, HashSet};

        let to_map = |fields: &Option<Vec<(String, Value)>>| -> HashMap<String, Value> {
            fields
                .as_ref()
                .map(|items| items.iter().map(|(k, v)| (k.clone(), v.clone())).collect())
                .unwrap_or_default()
        };

        let before_map = to_map(before_fields);
        let after_map = to_map(after_fields);

        let mut names: HashSet<String> = HashSet::new();
        names.extend(before_map.keys().cloned());
        names.extend(after_map.keys().cloned());

        let hash_index = self.database.storage().hash_index();

        for name in names {
            let before_value = before_map.get(&name);
            let after_value = after_map.get(&name);

            if before_value == after_value {
                continue;
            }

            if let Some(value) = before_value {
                if !matches!(value, Value::Null) {
                    hash_index.remove(entity, &name, value, entity_id)?;
                }
            }

            if let Some(value) = after_value {
                if !matches!(value, Value::Null) {
                    hash_index.insert(entity, &name, value, entity_id)?;
                }
            }
        }

        Ok(())
    }

    fn update_btree_indexes(
        &self,
        entity: &str,
        entity_id: [u8; 16],
        before_fields: &Option<Vec<(String, Value)>>,
        after_fields: &Option<Vec<(String, Value)>>,
    ) -> Result<(), Error> {
        use std::collections::{HashMap, HashSet};

        let Some(btree) = self.database.storage().btree_index() else {
            return Ok(());
        };

        let to_map = |fields: &Option<Vec<(String, Value)>>| -> HashMap<String, Value> {
            fields
                .as_ref()
                .map(|items| items.iter().map(|(k, v)| (k.clone(), v.clone())).collect())
                .unwrap_or_default()
        };

        let before_map = to_map(before_fields);
        let after_map = to_map(after_fields);

        let mut columns: HashSet<String> = HashSet::new();
        columns.extend(self.database.storage().btree_indexed_columns_for_entity(entity));

        let relations = self
            .database
            .catalog()
            .relations_to(entity)
            .map_err(|e| Error::Database(format!("failed to load relations: {}", e)))?;
        for relation in relations {
            columns.insert(relation.to_field.clone());
        }

        for column in columns {
            let before_value = before_map.get(&column);
            let after_value = after_map.get(&column);

            if before_value == after_value {
                continue;
            }

            if let Some(value) = before_value {
                if !matches!(value, Value::Null) {
                    btree.remove(entity, &column, value, entity_id)?;
                }
            }

            if let Some(value) = after_value {
                if !matches!(value, Value::Null) {
                    btree.insert(entity, &column, value, entity_id)?;
                }
            }
        }

        Ok(())
    }

    fn delete_columnar_row(
        &self,
        entity: &str,
        entity_id: [u8; 16],
    ) -> Result<(), Error> {
        let entity_def = self
            .database
            .catalog()
            .get_entity(entity)
            .map_err(|e| Error::Database(format!("failed to load entity: {}", e)))?
            .ok_or_else(|| Error::Database(format!("entity '{}' not found", entity)))?;

        let columns: Vec<&str> = entity_def
            .fields
            .iter()
            .map(|field| field.name.as_str())
            .collect();

        let projection = self
            .database
            .storage()
            .columnar()
            .projection(entity)
            .map_err(|e| Error::Database(format!("failed to load columnar projection: {}", e)))?;

        projection
            .delete_row(&entity_id, &columns)
            .map_err(|e| Error::Database(format!("failed to delete columnar row: {}", e)))?;

        Ok(())
    }
}

/// Format an ID as hex for error messages.
fn hex_id(id: &[u8; 16]) -> String {
    id.iter().map(|b| format!("{:02x}", b)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ormdb_core::catalog::{EntityDef, FieldDef, FieldType, ScalarType, SchemaBundle};

    fn setup_test_db() -> (tempfile::TempDir, Database) {
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

        (dir, db)
    }

    #[test]
    fn snapshot_read_sees_committed_mutation_via_watermark() {
        use ormdb_core::query::QueryExecutor;
        use ormdb_proto::GraphQuery;

        let (_dir, db) = setup_test_db();
        MutationExecutor::new(&db)
            .execute(&Mutation::insert("User", vec![FieldValue::new("name", "Zoe")]))
            .unwrap();

        // execute_snapshot reads as-of the watermark; if execute() did not advance
        // the watermark, this would read as-of 0 and see nothing.
        let qexec = QueryExecutor::new(db.storage(), db.catalog());
        let result = qexec.execute_snapshot(&GraphQuery::new("User")).unwrap();
        assert_eq!(
            result.entities[0].len(),
            1,
            "snapshot read must observe the committed insert (watermark advanced)"
        );
    }

    #[test]
    fn test_insert() {
        let (_dir, db) = setup_test_db();
        let executor = MutationExecutor::new(&db);

        let mutation = Mutation::insert(
            "User",
            vec![
                FieldValue::new("name", "Alice"),
                FieldValue::new("age", 30i32),
            ],
        );

        let result = executor.execute(&mutation).unwrap();
        assert_eq!(result.affected, 1);
        assert_eq!(result.inserted_ids.len(), 1);

        // Verify we can query the entity back
        let inserted_id = result.inserted_ids[0];
        let (_, record) = db.storage().get_latest(&inserted_id).unwrap().unwrap();
        let fields = ormdb_core::query::decode_entity(&record.data).unwrap();

        assert!(fields.iter().any(|(n, v)| n == "name" && *v == Value::String("Alice".into())));
        assert!(fields.iter().any(|(n, v)| n == "age" && *v == Value::Int32(30)));
    }

    #[test]
    fn test_update() {
        let (_dir, db) = setup_test_db();
        let executor = MutationExecutor::new(&db);

        // First insert
        let insert = Mutation::insert(
            "User",
            vec![
                FieldValue::new("name", "Alice"),
                FieldValue::new("age", 30i32),
            ],
        );
        let insert_result = executor.execute(&insert).unwrap();
        let id = insert_result.inserted_ids[0];

        // Then update
        let update = Mutation::update(
            "User",
            id,
            vec![FieldValue::new("name", "Alicia"), FieldValue::new("age", 31i32)],
        );
        let update_result = executor.execute(&update).unwrap();
        assert_eq!(update_result.affected, 1);

        // Verify changes
        let (_, record) = db.storage().get_latest(&id).unwrap().unwrap();
        let fields = ormdb_core::query::decode_entity(&record.data).unwrap();

        assert!(fields.iter().any(|(n, v)| n == "name" && *v == Value::String("Alicia".into())));
        assert!(fields.iter().any(|(n, v)| n == "age" && *v == Value::Int32(31)));
    }

    #[test]
    fn test_delete() {
        let (_dir, db) = setup_test_db();
        let executor = MutationExecutor::new(&db);

        // First insert
        let insert = Mutation::insert("User", vec![FieldValue::new("name", "Bob")]);
        let insert_result = executor.execute(&insert).unwrap();
        let id = insert_result.inserted_ids[0];

        // Then delete
        let delete = Mutation::delete("User", id);
        let delete_result = executor.execute(&delete).unwrap();
        assert_eq!(delete_result.affected, 1);

        // Verify deleted
        assert!(db.storage().get_latest(&id).unwrap().is_none());
    }

    #[test]
    fn test_delete_nonexistent() {
        let (_dir, db) = setup_test_db();
        let executor = MutationExecutor::new(&db);

        let delete = Mutation::delete("User", [0u8; 16]);
        let result = executor.execute(&delete).unwrap();
        assert_eq!(result.affected, 0);
    }

    #[test]
    fn test_upsert_insert() {
        let (_dir, db) = setup_test_db();
        let executor = MutationExecutor::new(&db);

        let upsert = Mutation::upsert(
            "User",
            None,
            vec![FieldValue::new("name", "Charlie")],
        );
        let result = executor.execute(&upsert).unwrap();
        assert_eq!(result.affected, 1);
        assert_eq!(result.inserted_ids.len(), 1);
    }

    #[test]
    fn test_upsert_update() {
        let (_dir, db) = setup_test_db();
        let executor = MutationExecutor::new(&db);

        // First insert
        let insert = Mutation::insert("User", vec![FieldValue::new("name", "Dave")]);
        let insert_result = executor.execute(&insert).unwrap();
        let id = insert_result.inserted_ids[0];

        // Then upsert (should update)
        let upsert = Mutation::upsert(
            "User",
            Some(id),
            vec![FieldValue::new("name", "David")],
        );
        let result = executor.execute(&upsert).unwrap();
        assert_eq!(result.affected, 1);
        assert!(result.inserted_ids.is_empty());

        // Verify
        let (_, record) = db.storage().get_latest(&id).unwrap().unwrap();
        let fields = ormdb_core::query::decode_entity(&record.data).unwrap();
        assert!(fields.iter().any(|(n, v)| n == "name" && *v == Value::String("David".into())));
    }

    #[test]
    fn test_batch() {
        let (_dir, db) = setup_test_db();
        let executor = MutationExecutor::new(&db);

        let batch = MutationBatch::from_mutations(vec![
            Mutation::insert("User", vec![FieldValue::new("name", "User1")]),
            Mutation::insert("User", vec![FieldValue::new("name", "User2")]),
            Mutation::insert("User", vec![FieldValue::new("name", "User3")]),
        ]);

        let result = executor.execute_batch(&batch).unwrap();
        assert_eq!(result.affected, 3);
        assert_eq!(result.inserted_ids.len(), 3);
    }
}
