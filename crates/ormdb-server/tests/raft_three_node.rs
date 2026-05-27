//! M6 prerequisite: a real 3-node Raft cluster elects a leader and replicates.
//!
//! Brings up three in-process Raft nodes, each with its own storage/Db and nng
//! transport on a distinct port, initializes a 3-node cluster, writes a mutation
//! through the leader, and verifies the committed entity is **replicated to the
//! followers' local storage** (checked at the storage layer, so no catalog needed
//! on followers).
//!
//! This validates election + log replication across separate Raft nodes — the
//! foundation for the M6 distributed fault-injection study. Cleanly-torn-down,
//! fault-injected, multi-process verification is the Jepsen harness.
//!
//! Runs only with `--features raft`.
#![cfg(feature = "raft")]

use std::sync::Arc;
use std::time::{Duration, Instant};

use ormdb_core::storage::StorageEngine;
use ormdb_proto::{FieldValue, Mutation, Value};
use ormdb_raft::types::{ClientRequest, ClientResponse};
use ormdb_raft::{RaftClusterManager, RaftConfig, RaftTlsConfig};
use ormdb_server::{make_apply_fn, Database};

fn free_port() -> u16 {
    std::net::TcpListener::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap()
        .port()
}

struct Node {
    db: Arc<Database>,
    manager: Arc<RaftClusterManager>,
}

async fn start_node(node_id: u64, addr: &str, members: &[(u64, String)], init: bool) -> Node {
    let dir = tempfile::tempdir().unwrap();
    // Keep the tempdir so the data path outlives the test (process exits at end).
    let path = dir.keep();
    let db = Arc::new(Database::open(&path).unwrap());

    let config = RaftConfig {
        node_id,
        raft_listen_addr: addr.to_string(),
        raft_advertise_addr: addr.to_string(),
        heartbeat_interval_ms: 100,
        election_timeout_min_ms: 300,
        election_timeout_max_ms: 600,
        snapshot_threshold: 1000,
        max_entries_per_append: 100,
        data_dir: path.join("raft"),
        tls: RaftTlsConfig::default(),
    };
    let apply_fn = make_apply_fn(db.clone());
    let db_arc = Arc::new(db.storage().db().clone());
    let manager = Arc::new(
        RaftClusterManager::new(config, db.storage_arc(), db_arc, Some(apply_fn))
            .await
            .expect("manager init"),
    );
    if init {
        manager
            .initialize_cluster(members.to_vec())
            .await
            .expect("cluster init");
    }
    Node { db, manager }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn three_node_cluster_replicates() {
    let addrs: Vec<String> = (0..3).map(|_| format!("127.0.0.1:{}", free_port())).collect();
    let members: Vec<(u64, String)> = (0..3).map(|i| (i as u64 + 1, addrs[i].clone())).collect();

    // Start all three nodes (transports must be up before initialization).
    let mut nodes = Vec::new();
    for i in 0..3 {
        let init = i == 0; // initialize the cluster from node 1
        nodes.push(start_node(i as u64 + 1, &addrs[i], &members, init).await);
    }

    // Wait for a leader to emerge.
    let mut leader_idx = None;
    let deadline = Instant::now() + Duration::from_secs(15);
    while Instant::now() < deadline {
        if let Some(idx) = (0..3).find(|&i| nodes[i].manager.is_leader()) {
            leader_idx = Some(idx);
            break;
        }
        tokio::time::sleep(Duration::from_millis(150)).await;
    }
    let leader_idx = leader_idx.expect("a leader should be elected within 15s");
    println!("leader is node {}", leader_idx + 1);

    // Write a mutation through the leader. The id is assigned up front (as the
    // request handler does) so every node applies the same id deterministically.
    let uid = StorageEngine::generate_id();
    let resp = tokio::time::timeout(
        Duration::from_secs(10),
        nodes[leader_idx].manager.write(ClientRequest::Mutate(Mutation::insert(
            "User",
            vec![
                FieldValue::new("id", Value::Uuid(uid)),
                FieldValue::new("name", "Alice"),
            ],
        ))),
    )
    .await
    .expect("write timed out")
    .expect("raft write");
    match resp {
        ClientResponse::MutationResult(r) => {
            assert_eq!(r.inserted_ids[0], uid, "leader applies the provided id");
        }
        other => panic!("expected MutationResult, got {other:?}"),
    }

    // The committed entity must replicate to every node's local storage, AND be
    // stamped with the SAME version timestamp (the Raft log index) everywhere —
    // deterministic application, so cluster snapshot reads agree.
    let mut versions = Vec::new();
    for i in 0..3 {
        let deadline = Instant::now() + Duration::from_secs(10);
        let mut version = None;
        while Instant::now() < deadline {
            if let Some((v, _)) = nodes[i].db.storage().get_latest(&uid).unwrap() {
                version = Some(v);
                break;
            }
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
        let v = version.unwrap_or_else(|| panic!("write must replicate to node {}", i + 1));
        versions.push(v);
    }
    assert!(
        versions.iter().all(|&v| v == versions[0]),
        "version timestamp must be identical across nodes (deterministic apply): {versions:?}"
    );

    println!(
        "three_node_cluster_replicates: ok (replicated to all nodes; version_ts={} on all)",
        versions[0]
    );
    // Native transport/core threads keep the process alive; exit after success.
    std::process::exit(0);
}
