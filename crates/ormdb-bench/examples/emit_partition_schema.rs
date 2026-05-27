//! Emit the rkyv-encoded SchemaBundle for the distributed partition study.
//!
//! The gateway `/schema` endpoint forwards raw bytes to `SchemaBundle::from_bytes`
//! (rkyv), so the study's HTTP client cannot send JSON — it POSTs these bytes.
//!
//! Usage: cargo run -p ormdb-bench --example emit_partition_schema -- <out.bin>

use ormdb_core::catalog::{EntityDef, FieldDef, FieldType, RelationDef, ScalarType, SchemaBundle};

fn main() {
    let out = std::env::args()
        .nth(1)
        .expect("usage: emit_partition_schema <out.bin>");

    let user = EntityDef::new("User", "id")
        .with_field(FieldDef::new("id", FieldType::Scalar(ScalarType::Uuid)))
        .with_field(FieldDef::new("name", FieldType::Scalar(ScalarType::String)))
        .with_field(FieldDef::new("gen", FieldType::Scalar(ScalarType::Int64)));

    let post = EntityDef::new("Post", "id")
        .with_field(FieldDef::new("id", FieldType::Scalar(ScalarType::Uuid)))
        .with_field(FieldDef::new("author_id", FieldType::Scalar(ScalarType::Uuid)))
        .with_field(FieldDef::new("gen", FieldType::Scalar(ScalarType::Int64)));

    let posts = RelationDef::one_to_many("posts", "User", "id", "Post", "author_id");

    let bundle = SchemaBundle::new(1)
        .with_entity(user)
        .with_entity(post)
        .with_relation(posts);

    let bytes = bundle.to_bytes().expect("serialize schema");
    std::fs::write(&out, &bytes).expect("write schema file");
    eprintln!("wrote {} bytes to {}", bytes.len(), out);
}
