//! M0 integrity gate: demonstrate the *fractured graph read* anomaly.
//!
//! A graph fetch assembles a root entity and its related entities from
//! *independent* latest-version reads. The root is read via
//! `StorageEngine::get_latest` (mirroring `QueryExecutor::execute_plan` ->
//! `fetch_entities`, `crates/ormdb-core/src/query/executor.rs:495`) and the
//! related entities are read later via `get_latest_batch`
//! (mirroring `resolve_includes`, `executor.rs:520`). No shared read snapshot
//! is taken between the two phases.
//!
//! Consequence: under a concurrent committed write, a single graph fetch can
//! return an object graph that *never existed as a committed state* — a parent
//! observed at generation N paired with a child observed at generation N+1.
//!
//! These tests assert the anomaly is *present today* (documenting the bug). After
//! milestone M4 (snapshot-consistent graph fetch) the executor will read both
//! phases as-of a single timestamp; the companion fix test will then assert the
//! anomaly is *gone*. The final block of the storage-level test already shows the
//! fix primitive (`get_at`) reconstructs a consistent cut.

use ormdb_bench::TestContext;
use ormdb_core::query::{decode_entity, encode_entity};
use ormdb_core::storage::{Record, StorageEngine, VersionedKey};
use ormdb_proto::Value;

/// Fields for a `User` whose `name` encodes the current generation.
fn user_fields(id: [u8; 16], generation: i64) -> Vec<(String, Value)> {
    vec![
        ("id".into(), Value::Uuid(id)),
        ("name".into(), Value::String(format!("gen{generation}"))),
        ("email".into(), Value::String("u@example.com".into())),
        ("age".into(), Value::Int32(30)),
        ("status".into(), Value::String("active".into())),
    ]
}

/// Fields for a `Post` whose `title` encodes the current generation and whose
/// `author_id` points at its owning `User`.
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

fn put(storage: &StorageEngine, entity_type: &str, id: [u8; 16], version_ts: u64, fields: Vec<(String, Value)>) {
    let data = encode_entity(&fields).unwrap();
    storage
        .put_typed(entity_type, VersionedKey::new(id, version_ts), Record::new(data))
        .unwrap();
}

fn string_field(fields: &[(String, Value)], name: &str) -> String {
    fields
        .iter()
        .find(|(k, _)| k == name)
        .and_then(|(_, v)| match v {
            Value::String(s) => Some(s.clone()),
            _ => None,
        })
        .unwrap_or_else(|| panic!("missing string field {name}"))
}

/// Atomically advance both entities to a new generation, preserving the
/// cross-entity invariant `User.name == Post.title` on disk.
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

/// Storage-level reproduction that mirrors the two-phase read inside a single
/// `execute_plan` call.
#[test]
fn fractured_graph_read_is_observable_at_storage_layer() {
    let ctx = TestContext::with_schema();
    let storage = &ctx.storage;
    let user = StorageEngine::generate_id();
    let post = StorageEngine::generate_id();

    // gen0: invariant holds on disk (User.name == Post.title == "gen0").
    put(storage, "User", user, 1_000, user_fields(user, 0));
    put(storage, "Post", post, 1_000, post_fields(post, user, 0));

    // Phase 1 — root read (executor.rs:495, fetch_entities -> get_latest).
    let (root_version, user_rec) = storage.get_latest(&user).unwrap().unwrap();
    let user_gen = string_field(&decode_entity(&user_rec.data).unwrap(), "name");
    assert_eq!(user_gen, "gen0");

    // A concurrent transaction commits BETWEEN the two read phases, atomically
    // advancing both entities to gen1. The invariant still holds on disk.
    commit_generation(storage, user, post, 2_000, 1);

    // Phase 2 — include read (executor.rs:520, resolve_includes -> get_latest_batch).
    let (child_version, post_rec) = storage.get_latest_batch(&[post]).unwrap()[0].clone().unwrap();
    let post_gen = string_field(&decode_entity(&post_rec.data).unwrap(), "title");
    assert_eq!(post_gen, "gen1");

    // The assembled graph pairs a gen0 parent with a gen1 child: a combination
    // that never existed as a committed state. THIS is the fractured graph read.
    assert!(child_version > root_version, "child read at a later version than root");
    assert_ne!(
        user_gen, post_gen,
        "fractured graph read: root observed gen0 but related entity observed gen1"
    );

    // The M4 fix primitive already exists in the engine: reading BOTH entities
    // as-of the root's version reconstructs a consistent cut (invariant restored).
    let (_, user_at) = storage.get_at(&user, root_version).unwrap().unwrap();
    let (_, post_at) = storage.get_at(&post, root_version).unwrap().unwrap();
    assert_eq!(
        string_field(&decode_entity(&user_at.data).unwrap(), "name"),
        string_field(&decode_entity(&post_at.data).unwrap(), "title"),
        "snapshot read (get_at) at the root version must be a consistent cut"
    );
}

/// Executor-level reproduction of the lazy-loading / N+1 pattern: the root and
/// the relation are loaded by two separate graph fetches, exactly as a lazy ORM
/// does. A write that commits in between yields a torn client-assembled graph.
#[test]
fn fractured_graph_read_via_lazy_loading_through_executor() {
    use ormdb_proto::{FilterExpr, GraphQuery};

    let ctx = TestContext::with_schema();
    let storage = &ctx.storage;
    let user = StorageEngine::generate_id();
    let post = StorageEngine::generate_id();

    put(storage, "User", user, 1_000, user_fields(user, 0));
    put(storage, "Post", post, 1_000, post_fields(post, user, 0));

    // Query 1: the ORM loads the root entity.
    let q_user = GraphQuery::new("User").with_filter(FilterExpr::eq("id", Value::Uuid(user)).into());
    let r1 = ctx.executor().execute(&q_user).unwrap();
    let user_gen = match &r1.entities[0].column("name").unwrap().values[0] {
        Value::String(s) => s.clone(),
        other => panic!("unexpected name value: {other:?}"),
    };
    assert_eq!(user_gen, "gen0");

    // A write commits before the ORM lazily resolves the relation.
    commit_generation(storage, user, post, 2_000, 1);

    // Query 2: the ORM lazily loads the user's posts (the "+1" of N+1).
    let q_posts =
        GraphQuery::new("Post").with_filter(FilterExpr::eq("author_id", Value::Uuid(user)).into());
    let r2 = ctx.executor().execute(&q_posts).unwrap();
    let post_gen = match &r2.entities[0].column("title").unwrap().values[0] {
        Value::String(s) => s.clone(),
        other => panic!("unexpected title value: {other:?}"),
    };
    assert_eq!(post_gen, "gen1");

    assert_ne!(
        user_gen, post_gen,
        "lazy-loaded relation observed gen1 while the root was gen0: fractured graph"
    );
}
