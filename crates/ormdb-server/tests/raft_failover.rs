//! M6 fault injection: leader failure -> re-election -> continued consistency.
//!
//! The `kill` nemesis: a 3-node cluster replicates a write, the leader is then
//! shut down (simulating node failure), the surviving two re-elect a leader and
//! remain available, a second write succeeds on the new leader, and BOTH writes
//! are present on the surviving nodes (durability + availability across failover
//! with a 2/3 quorum).
//!
//! In-process (nng transport); network partitions need separate processes and are
//! the Jepsen harness's job. Runs only with `--features raft`.
#![cfg(feature = "raft")]

use std::sync::Arc;
use std::time::{Duration, Instant};

use ormdb_core::storage::StorageEngine;
use ormdb_proto::{FieldValue, Mutation, Value};
use ormdb_raft::types::ClientRequest;
use ormdb_raft::{RaftClusterManager, RaftConfig, RaftTlsConfig};
use ormdb_server::{make_apply_fn, Database};

fn free_port() -> u16 {
    std::net::TcpListener::bind("127.0.0.1:0").unwrap().local_addr().unwrap().port()
}

struct Node {
    db: Arc<Database>,
    manager: Arc<RaftClusterManager>,
    alive: bool,
}

async fn start_node(node_id: u64, addr: &str, members: &[(u64, String)], init: bool) -> Node {
    let path = tempfile::tempdir().unwrap().keep();
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
        manager.initialize_cluster(members.to_vec()).await.expect("cluster init");
    }
    Node { db, manager, alive: true }
}

async fn find_leader(nodes: &[Node], timeout: Duration) -> Option<usize> {
    let deadline = Instant::now() + timeout;
    while Instant::now() < deadline {
        if let Some(i) = (0..nodes.len()).find(|&i| nodes[i].alive && nodes[i].manager.is_leader()) {
            return Some(i);
        }
        tokio::time::sleep(Duration::from_millis(150)).await;
    }
    None
}

async fn insert_via(node: &Node, name: &str) -> [u8; 16] {
    let uid = StorageEngine::generate_id();
    tokio::time::timeout(
        Duration::from_secs(10),
        node.manager.write(ClientRequest::Mutate(Mutation::insert(
            "User",
            vec![FieldValue::new("id", Value::Uuid(uid)), FieldValue::new("name", name)],
        ))),
    )
    .await
    .expect("write timed out")
    .expect("raft write");
    uid
}

async fn await_replicated(node: &Node, uid: &[u8; 16], timeout: Duration) -> bool {
    let deadline = Instant::now() + timeout;
    while Instant::now() < deadline {
        if node.db.storage().get_latest(uid).unwrap().is_some() {
            return true;
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    false
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn leader_failure_reelects_and_stays_consistent() {
    let addrs: Vec<String> = (0..3).map(|_| format!("127.0.0.1:{}", free_port())).collect();
    let members: Vec<(u64, String)> = (0..3).map(|i| (i as u64 + 1, addrs[i].clone())).collect();

    let mut nodes = Vec::new();
    for i in 0..3 {
        nodes.push(start_node(i as u64 + 1, &addrs[i], &members, i == 0).await);
    }

    let leader = find_leader(&nodes, Duration::from_secs(15)).await.expect("initial leader");
    println!("initial leader: node {}", leader + 1);

    // First write, replicated cluster-wide.
    let uid1 = insert_via(&nodes[leader], "Alice").await;

    // Kill the leader (node failure).
    nodes[leader].manager.shutdown().await.ok();
    nodes[leader].alive = false;
    println!("killed leader node {}", leader + 1);

    // Survivors must re-elect a leader (2/3 quorum).
    let new_leader = find_leader(&nodes, Duration::from_secs(20)).await.expect("re-election");
    assert_ne!(new_leader, leader, "a surviving node becomes the new leader");
    println!("new leader after failover: node {}", new_leader + 1);

    // Second write succeeds on the new leader (cluster still available).
    let uid2 = insert_via(&nodes[new_leader], "Bob").await;

    // Both writes are present on every surviving node.
    for i in 0..3 {
        if !nodes[i].alive {
            continue;
        }
        assert!(await_replicated(&nodes[i], &uid1, Duration::from_secs(10)).await,
                "pre-failover write durable on node {}", i + 1);
        assert!(await_replicated(&nodes[i], &uid2, Duration::from_secs(10)).await,
                "post-failover write present on node {}", i + 1);
    }

    println!("leader_failure_reelects_and_stays_consistent: ok");
    std::process::exit(0);
}
