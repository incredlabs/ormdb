//! M2 experiment: measure the *fractured graph read* rate under concurrency.
//!
//! A writer thread atomically advances a global "generation" tag stamped on every
//! `User.name` and every `Post.title`. Reader threads repeatedly issue a graph
//! fetch `User { include posts }` and inspect the assembled graph: if it contains
//! more than one distinct generation tag, the graph was torn across a concurrent
//! commit — a fractured graph read.
//!
//! We compare the read-committed path (`execute`) against the snapshot path
//! (`execute_snapshot`, milestone M4). The snapshot path must report zero
//! fractures.
//!
//! Run: `cargo run -p ormdb-bench --release --example anomaly_rate`

use std::collections::HashSet;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, Instant};

use ormdb_bench::TestContext;
use ormdb_core::query::encode_entity;
use ormdb_core::storage::key::current_timestamp;
use ormdb_core::storage::{Record, StorageEngine, VersionedKey};
use ormdb_proto::{GraphQuery, QueryResult, RelationInclude, Value};

/// Number of (User, Post) pairs. More entities widen the read window between the
/// root scan and the include scan, which is where the tear occurs.
const NUM_PAIRS: usize = 200;

fn user_fields(id: [u8; 16], generation: u64) -> Vec<(String, Value)> {
    vec![
        ("id".into(), Value::Uuid(id)),
        ("name".into(), Value::String(format!("gen{generation}"))),
        ("email".into(), Value::String("u@example.com".into())),
        ("age".into(), Value::Int32(30)),
        ("status".into(), Value::String("active".into())),
    ]
}

fn post_fields(id: [u8; 16], author_id: [u8; 16], generation: u64) -> Vec<(String, Value)> {
    vec![
        ("id".into(), Value::Uuid(id)),
        ("author_id".into(), Value::Uuid(author_id)),
        ("title".into(), Value::String(format!("gen{generation}"))),
        ("content".into(), Value::String("body".into())),
        ("views".into(), Value::Int64(0)),
        ("published".into(), Value::Bool(true)),
    ]
}

fn record(fields: Vec<(String, Value)>) -> Record {
    Record::new(encode_entity(&fields).unwrap())
}

/// Atomically stamp every entity with `generation`, all at one version timestamp.
fn write_generation(storage: &StorageEngine, pairs: &[([u8; 16], [u8; 16])], generation: u64) {
    let ts = current_timestamp();
    let mut txn = storage.transaction();
    for (user, post) in pairs {
        txn.put_typed("User", VersionedKey::new(*user, ts), record(user_fields(*user, generation)));
        txn.put_typed("Post", VersionedKey::new(*post, ts), record(post_fields(*post, *user, generation)));
    }
    // commit_versioned re-stamps every write with one monotonic commit timestamp
    // and advances the read watermark only once the data is visible — so a reader
    // that captures the watermark sees a single consistent generation.
    txn.commit_versioned().unwrap();
}

/// Collect the distinct generation tags present across all `User.name` and
/// `Post.title` values in a graph-fetch result.
fn distinct_generations(result: &QueryResult) -> HashSet<String> {
    let mut gens = HashSet::new();
    for block in &result.entities {
        let col = match block.entity.as_str() {
            "User" => "name",
            "Post" => "title",
            _ => continue,
        };
        if let Some(column) = block.column(col) {
            for v in &column.values {
                if let Value::String(s) = v {
                    gens.insert(s.clone());
                }
            }
        }
    }
    gens
}

#[derive(Default)]
struct Counts {
    reads: u64,
    fractures: u64,
}

fn run_mode(ctx: &TestContext, pairs: &[([u8; 16], [u8; 16])], readers: usize, snapshot: bool, dur: Duration) -> Counts {
    let storage = &ctx.storage;
    let catalog = &ctx.catalog;
    let stop = AtomicBool::new(false);
    let generation = AtomicU64::new(0);
    let reads = AtomicU64::new(0);
    let fractures = AtomicU64::new(0);

    std::thread::scope(|scope| {
        // Writer: hammer the generation counter.
        scope.spawn(|| {
            while !stop.load(Ordering::Relaxed) {
                let g = generation.fetch_add(1, Ordering::Relaxed) + 1;
                write_generation(storage, pairs, g);
            }
        });

        // Readers: graph-fetch and check for a torn graph.
        for _ in 0..readers {
            scope.spawn(|| {
                let query = GraphQuery::new("User").include(RelationInclude::new("posts"));
                let executor = ormdb_core::query::QueryExecutor::new(storage, catalog);
                while !stop.load(Ordering::Relaxed) {
                    let result = if snapshot {
                        executor.execute_snapshot(&query)
                    } else {
                        executor.execute(&query)
                    };
                    if let Ok(result) = result {
                        reads.fetch_add(1, Ordering::Relaxed);
                        if distinct_generations(&result).len() > 1 {
                            fractures.fetch_add(1, Ordering::Relaxed);
                        }
                    }
                }
            });
        }

        std::thread::sleep(dur);
        stop.store(true, Ordering::Relaxed);
    });

    Counts { reads: reads.load(Ordering::Relaxed), fractures: fractures.load(Ordering::Relaxed) }
}

fn main() {
    let dur = Duration::from_millis(
        std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(2000),
    );

    // Build schema and seed gen0.
    let ctx = TestContext::with_schema();
    let pairs: Vec<([u8; 16], [u8; 16])> = (0..NUM_PAIRS)
        .map(|_| (StorageEngine::generate_id(), StorageEngine::generate_id()))
        .collect();
    write_generation(&ctx.storage, &pairs, 0);

    println!(
        "fractured graph read rate  |  {} (User,Post) pairs, {} ms per cell\n",
        NUM_PAIRS,
        dur.as_millis()
    );
    println!("{:<10} {:>10} {:>12} {:>10}   {:>10} {:>12} {:>10}",
             "readers", "rc_reads", "rc_fractures", "rc_rate", "snap_reads", "snap_fract", "snap_rate");

    for &readers in &[1usize, 2, 4, 8] {
        let start = Instant::now();
        let rc = run_mode(&ctx, &pairs, readers, false, dur);
        let snap = run_mode(&ctx, &pairs, readers, true, dur);
        let rc_rate = 100.0 * rc.fractures as f64 / rc.reads.max(1) as f64;
        let snap_rate = 100.0 * snap.fractures as f64 / snap.reads.max(1) as f64;
        println!(
            "{:<10} {:>10} {:>12} {:>9.2}%   {:>10} {:>12} {:>9.2}%   ({:?})",
            readers, rc.reads, rc.fractures, rc_rate, snap.reads, snap.fractures, snap_rate, start.elapsed()
        );
    }

    println!("\nrc   = read-committed (execute): root and includes read independently");
    println!("snap = snapshot (execute_snapshot, M4): root + includes share one cut");
}
