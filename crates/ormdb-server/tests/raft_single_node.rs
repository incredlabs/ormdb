//! M5 end-to-end: a single-node Raft cluster drives the full write + read path.
//!
//! Brings up a real (single-node) Raft cluster, writes a mutation through
//! consensus (`RaftClusterManager::write` -> log -> commit -> apply callback ->
//! local storage), establishes a linearizable read point via ReadIndex
//! (`ensure_linearizable`), and reads the result back from local storage.
//!
//! Uses the async manager APIs (the `*_blocking` variants are for the synchronous
//! request handler; calling them on the test's runtime thread would deadlock).
//!
//! Note on teardown: openraft's core and the nng transport spawn native threads
//! that are not joined when the runtime drops, so the process does not exit on its
//! own after the test body completes. All assertions run first; we then exit
//! explicitly. (Multi-node, cleanly-torn-down verification is M6's Jepsen harness.)
//!
//! Runs only with `--features raft`.
#![cfg(feature = "raft")]

use std::sync::Arc;
use std::time::Duration;

use ormdb_core::catalog::{EntityDef, FieldDef, FieldType, ScalarType, SchemaBundle};
use ormdb_core::query::QueryExecutor;
use ormdb_proto::{FieldValue, GraphQuery, Mutation, Value};
use ormdb_raft::types::ClientRequest;
use ormdb_raft::{RaftClusterManager, RaftConfig, RaftTlsConfig};
use ormdb_server::{make_apply_fn, Database};

fn free_port() -> u16 {
    std::net::TcpListener::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap()
        .port()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn single_node_raft_write_and_linearizable_read() {
    let dir = tempfile::tempdir().unwrap();
    let database = Arc::new(Database::open(dir.path()).unwrap());
    let schema = SchemaBundle::new(1).with_entity(
        EntityDef::new("User", "id")
            .with_field(FieldDef::new("id", FieldType::Scalar(ScalarType::Uuid)))
            .with_field(FieldDef::new("name", FieldType::Scalar(ScalarType::String))),
    );
    database.catalog().apply_schema(schema).unwrap();

    let addr = format!("127.0.0.1:{}", free_port());
    let config = RaftConfig {
        node_id: 1,
        raft_listen_addr: addr.clone(),
        raft_advertise_addr: addr.clone(),
        heartbeat_interval_ms: 100,
        election_timeout_min_ms: 200,
        election_timeout_max_ms: 400,
        snapshot_threshold: 1000,
        max_entries_per_append: 100,
        data_dir: dir.path().join("raft"),
        tls: RaftTlsConfig::default(),
    };

    let apply_fn = make_apply_fn(database.clone());
    let db_arc = Arc::new(database.storage().db().clone());
    let manager = RaftClusterManager::new(config, database.storage_arc(), db_arc, Some(apply_fn))
        .await
        .expect("manager init");
    manager
        .initialize_cluster(vec![(1, addr)])
        .await
        .expect("cluster init");

    // Wait for this node to win the election.
    let mut became_leader = false;
    for _ in 0..50 {
        if manager.is_leader() {
            became_leader = true;
            break;
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    assert!(became_leader, "single node should become leader");

    // Write through Raft consensus (committed + applied via the apply callback).
    let resp = tokio::time::timeout(
        Duration::from_secs(10),
        manager.write(ClientRequest::Mutate(Mutation::insert(
            "User",
            vec![FieldValue::new("name", "Alice")],
        ))),
    )
    .await
    .expect("raft write timed out")
    .expect("raft write");
    assert!(!resp.is_error(), "raft write should succeed: {resp:?}");

    // ReadIndex, then read local storage — must observe the committed write.
    tokio::time::timeout(Duration::from_secs(10), manager.ensure_linearizable())
        .await
        .expect("ensure_linearizable timed out")
        .expect("linearizable read point");

    let executor = QueryExecutor::new(database.storage(), database.catalog());
    let result = executor.execute(&GraphQuery::new("User")).unwrap();
    assert_eq!(result.entities[0].len(), 1, "the replicated user is readable");
    match &result.entities[0].column("name").unwrap().values[0] {
        Value::String(s) => assert_eq!(s, "Alice"),
        other => panic!("unexpected name value: {other:?}"),
    }

    // All assertions passed. Best-effort graceful shutdown, then exit explicitly
    // because native transport/core threads keep the process alive otherwise.
    let _ = tokio::time::timeout(Duration::from_secs(5), manager.shutdown()).await;
    println!("single_node_raft_write_and_linearizable_read: ok");
    std::process::exit(0);
}
