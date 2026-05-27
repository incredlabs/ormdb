//! M4: snapshot-consistent graph fetch eliminates the fractured graph read.
//!
//! `QueryExecutor::execute_as_of(query, read_ts)` materializes the root entity
//! AND every relation include as-of a single `read_ts`, so the assembled graph
//! corresponds to one commit cut. This test pairs with the M0 anomaly tests
//! (`fractured_graph_read.rs`): the same data that yields a torn graph under the
//! read-committed path yields a graph-atomic result under the snapshot path.

use ormdb_bench::TestContext;
use ormdb_core::query::encode_entity;
use ormdb_core::storage::{Record, StorageEngine, VersionedKey};
use ormdb_proto::{GraphQuery, RelationInclude, Value};

fn user_fields(id: [u8; 16], generation: i64) -> Vec<(String, Value)> {
    vec![
        ("id".into(), Value::Uuid(id)),
        ("name".into(), Value::String(format!("gen{generation}"))),
        ("email".into(), Value::String("u@example.com".into())),
        ("age".into(), Value::Int32(30)),
        ("status".into(), Value::String("active".into())),
    ]
}

fn post_fields(id: [u8; 16], author_id: [u8; 16], generation: i64) -> Vec<(String, Value)> {
    vec![
        ("id".into(), Value::Uuid(id)),
        ("author_id".into(), Value::Uuid(author_id)),
        ("title".into(), Value::String(format!("gen{generation}"))),
        ("content".into(), Value::String("body".into())),
        ("views".into(), Value::Int64(0)),
        ("published".into(), Value::Bool(true)),
    ]
}

fn rec(fields: Vec<(String, Value)>) -> Record {
    Record::new(encode_entity(&fields).unwrap())
}

fn put(storage: &StorageEngine, entity_type: &str, id: [u8; 16], version_ts: u64, fields: Vec<(String, Value)>) {
    let data = encode_entity(&fields).unwrap();
    storage
        .put_typed(entity_type, VersionedKey::new(id, version_ts), Record::new(data))
        .unwrap();
}

fn commit_generation(storage: &StorageEngine, user: [u8; 16], post: [u8; 16], version_ts: u64, generation: i64) {
    let mut txn = storage.transaction();
    txn.put_typed(
        "User",
        VersionedKey::new(user, version_ts),
        Record::new(encode_entity(&user_fields(user, generation)).unwrap()),
    );
    txn.put_typed(
        "Post",
        VersionedKey::new(post, version_ts),
        Record::new(encode_entity(&post_fields(post, user, generation)).unwrap()),
    );
    txn.commit().unwrap();
}

/// First string value of `column` in the block named `entity`.
fn block_str(result: &ormdb_proto::QueryResult, entity: &str, column: &str) -> String {
    let block = result
        .entities
        .iter()
        .find(|b| b.entity == entity)
        .unwrap_or_else(|| panic!("no entity block named {entity}"));
    match &block.column(column).unwrap().values[0] {
        Value::String(s) => s.clone(),
        other => panic!("{entity}.{column} not a string: {other:?}"),
    }
}

/// A graph fetch as-of a timestamp returns a single, consistent commit cut for
/// the root AND its includes — never the torn (parent gen0, child gen1) pair.
#[test]
fn snapshot_graph_fetch_is_graph_atomic() {
    let ctx = TestContext::with_schema();
    let storage = &ctx.storage;
    let user = StorageEngine::generate_id();
    let post = StorageEngine::generate_id();

    // gen0 committed at ts=1000, gen1 committed at ts=2000 (both atomic on disk).
    put(storage, "User", user, 1_000, user_fields(user, 0));
    put(storage, "Post", post, 1_000, post_fields(post, user, 0));
    commit_generation(storage, user, post, 2_000, 1);

    let query = GraphQuery::new("User").include(RelationInclude::new("posts"));

    // Snapshot as-of ts=1500 (between the two commits): everything is gen0.
    let r_old = ctx.executor().execute_as_of(&query, 1_500).unwrap();
    assert_eq!(block_str(&r_old, "User", "name"), "gen0");
    assert_eq!(block_str(&r_old, "Post", "title"), "gen0", "include must share the root's cut");
    assert_eq!(r_old.edges.iter().map(|e| e.edges.len()).sum::<usize>(), 1, "user->post edge present");

    // Snapshot as-of ts=2500 (after both commits): everything is gen1.
    let r_new = ctx.executor().execute_as_of(&query, 2_500).unwrap();
    assert_eq!(block_str(&r_new, "User", "name"), "gen1");
    assert_eq!(block_str(&r_new, "Post", "title"), "gen1");

    // Crucially, no snapshot ever yields the torn (root gen0, child gen1) graph
    // that the read-committed path produces in fractured_graph_read.rs.
    for (root, child) in [(&r_old, "gen0"), (&r_new, "gen1")] {
        assert_eq!(block_str(root, "User", "name"), child);
        assert_eq!(block_str(root, "Post", "title"), child);
    }
}

/// Entities created after the snapshot timestamp are invisible; the relation
/// include does not observe writes that commit after the read point.
#[test]
fn snapshot_excludes_writes_after_read_ts() {
    let ctx = TestContext::with_schema();
    let storage = &ctx.storage;
    let user = StorageEngine::generate_id();
    let post1 = StorageEngine::generate_id();
    let post2 = StorageEngine::generate_id();

    put(storage, "User", user, 1_000, user_fields(user, 0));
    put(storage, "Post", post1, 1_000, post_fields(post1, user, 0));
    // A second post is added later, at ts=2000.
    put(storage, "Post", post2, 2_000, post_fields(post2, user, 0));

    let query = GraphQuery::new("User").include(RelationInclude::new("posts"));

    // As-of ts=1500: only the first post is visible.
    let r = ctx.executor().execute_as_of(&query, 1_500).unwrap();
    let post_block = r.entities.iter().find(|b| b.entity == "Post").unwrap();
    assert_eq!(post_block.len(), 1, "post created after read_ts must be excluded");

    // As-of ts=2500: both posts are visible.
    let r2 = ctx.executor().execute_as_of(&query, 2_500).unwrap();
    let post_block2 = r2.entities.iter().find(|b| b.entity == "Post").unwrap();
    assert_eq!(post_block2.len(), 2);
}

/// With `commit_versioned`, the read watermark anchors a sound snapshot: a fetch
/// as-of the old watermark sees one consistent generation for root AND include,
/// even though a newer generation has since committed.
#[test]
fn watermark_snapshot_is_consistent_under_versioned_commits() {
    let ctx = TestContext::with_schema();
    let storage = &ctx.storage;
    let user = StorageEngine::generate_id();
    let post = StorageEngine::generate_id();

    // gen0 via commit_versioned advances the watermark.
    let mut t0 = storage.transaction();
    t0.put_typed("User", VersionedKey::now(user), rec(user_fields(user, 0)));
    t0.put_typed("Post", VersionedKey::now(post), rec(post_fields(post, user, 0)));
    t0.commit_versioned().unwrap();
    let w0 = storage.read_watermark();

    // gen1 via commit_versioned gets a strictly greater timestamp.
    let mut t1 = storage.transaction();
    t1.put_typed("User", VersionedKey::now(user), rec(user_fields(user, 1)));
    t1.put_typed("Post", VersionedKey::now(post), rec(post_fields(post, user, 1)));
    let ts1 = t1.commit_versioned().unwrap();
    assert!(ts1 > w0, "commit timestamps must be monotonic");
    assert!(storage.read_watermark() > w0);

    // A fetch as-of the old watermark sees gen0 for both root and include.
    let query = GraphQuery::new("User").include(RelationInclude::new("posts"));
    let r = ctx.executor().execute_as_of(&query, w0).unwrap();
    assert_eq!(block_str(&r, "User", "name"), "gen0");
    assert_eq!(block_str(&r, "Post", "title"), "gen0");
}

/// `get_as_of` respects tombstones: a deletion at-or-before the read point hides
/// the entity (it does not resurrect an older live version).
#[test]
fn get_as_of_respects_tombstones() {
    let ctx = TestContext::with_schema();
    let storage = &ctx.storage;
    let post = StorageEngine::generate_id();

    put(storage, "Post", post, 1_000, post_fields(post, StorageEngine::generate_id(), 0));
    // Tombstone committed at ts=2000.
    storage
        .put_typed("Post", VersionedKey::new(post, 2_000), Record::tombstone())
        .unwrap();

    assert!(storage.get_as_of(&post, 1_500).unwrap().is_some(), "live before delete");
    assert!(storage.get_as_of(&post, 2_500).unwrap().is_none(), "hidden at/after delete");
}
