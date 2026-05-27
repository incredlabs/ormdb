//! M5: a committed Raft mutation applies to local storage.
//!
//! `make_apply_fn` is the callback the Raft state machine invokes for each
//! committed log entry (see `raft_apply.rs`). This test exercises that bridge
//! directly — simulating a committed `ClientRequest` — and verifies the mutation
//! is visible in local storage, i.e. consensus now drives the real write path.
//!
//! Runs only with `--features raft`.
#![cfg(feature = "raft")]

use std::sync::Arc;

use ormdb_core::catalog::{EntityDef, FieldDef, FieldType, ScalarType, SchemaBundle};
use ormdb_core::query::QueryExecutor;
use ormdb_proto::{FieldValue, GraphQuery, Mutation, Value};
use ormdb_raft::types::{ClientRequest, ClientResponse};
use ormdb_server::{make_apply_fn, Database};

#[test]
fn raft_apply_fn_applies_mutation_to_storage() {
    let dir = tempfile::tempdir().unwrap();
    let database = Arc::new(Database::open(dir.path()).unwrap());

    let schema = SchemaBundle::new(1).with_entity(
        EntityDef::new("User", "id")
            .with_field(FieldDef::new("id", FieldType::Scalar(ScalarType::Uuid)))
            .with_field(FieldDef::new("name", FieldType::Scalar(ScalarType::String))),
    );
    database.catalog().apply_schema(schema).unwrap();

    // The callback the Raft state machine runs for each committed entry.
    let apply_fn = make_apply_fn(database.clone());

    // Simulate applying a committed ClientRequest on this node.
    let req = ClientRequest::Mutate(Mutation::insert(
        "User",
        vec![FieldValue::new("name", "Alice")],
    ));
    match apply_fn(&req).expect("apply_fn should succeed") {
        ClientResponse::MutationResult(r) => {
            assert_eq!(r.affected, 1, "one row inserted");
            assert!(!r.inserted_ids.is_empty(), "an id was assigned");
        }
        other => panic!("expected MutationResult, got {other:?}"),
    }

    // The applied mutation must be visible in local storage.
    let executor = QueryExecutor::new(database.storage(), database.catalog());
    let result = executor.execute(&GraphQuery::new("User")).unwrap();
    assert_eq!(result.entities[0].len(), 1, "the inserted user is readable");
    match &result.entities[0].column("name").unwrap().values[0] {
        Value::String(s) => assert_eq!(s, "Alice"),
        other => panic!("unexpected name value: {other:?}"),
    }
}

/// A Noop request applies cleanly without touching storage.
#[test]
fn raft_apply_fn_handles_noop() {
    let dir = tempfile::tempdir().unwrap();
    let database = Arc::new(Database::open(dir.path()).unwrap());
    let apply_fn = make_apply_fn(database);

    assert!(matches!(
        apply_fn(&ClientRequest::Noop).unwrap(),
        ClientResponse::NoopResult
    ));
}
