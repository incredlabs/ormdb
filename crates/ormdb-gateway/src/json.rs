//! JSON request and response types for the HTTP gateway.

use serde::Serialize;
use ormdb_proto::value::Value;
use ormdb_proto::result::{QueryResult, MutationResult};
use ormdb_proto::replication::ReplicationStatus;

/// Generic success response wrapper.
#[derive(Debug, Serialize)]
pub struct SuccessResponse<T: Serialize> {
    /// Success flag.
    pub success: bool,
    /// Response data.
    pub data: T,
}

impl<T: Serialize> SuccessResponse<T> {
    /// Create a new success response.
    pub fn new(data: T) -> Self {
        Self {
            success: true,
            data,
        }
    }
}

/// Health check response.
#[derive(Debug, Serialize)]
pub struct HealthResponse {
    /// Health status.
    pub status: String,
    /// Gateway version.
    pub version: String,
    /// Whether connected to ORMDB server.
    pub ormdb_connected: bool,
}

/// Query response.
#[derive(Debug, Serialize)]
pub struct QueryResponse {
    /// Success flag.
    pub success: bool,
    /// Query result data.
    pub data: QueryResultJson,
    /// Metadata about the query.
    pub meta: QueryMeta,
}

/// Query result in JSON-friendly format.
#[derive(Debug, Serialize)]
pub struct QueryResultJson {
    /// Entity blocks (rows).
    pub entities: Vec<EntityBlockJson>,
    /// Edge blocks (relationships).
    pub edges: Vec<EdgeBlockJson>,
}

/// Entity block in JSON-friendly format.
#[derive(Debug, Serialize)]
pub struct EntityBlockJson {
    /// Entity type name.
    pub entity: String,
    /// Rows as JSON objects.
    pub rows: Vec<serde_json::Value>,
}

/// Edge block in JSON-friendly format.
#[derive(Debug, Serialize)]
pub struct EdgeBlockJson {
    /// Relation name.
    pub relation: String,
    /// Edges.
    pub edges: Vec<EdgeJson>,
}

/// Edge in JSON-friendly format.
#[derive(Debug, Serialize)]
pub struct EdgeJson {
    /// Source entity ID.
    pub from_id: String,
    /// Target entity ID.
    pub to_id: String,
}

/// Query metadata.
#[derive(Debug, Serialize)]
pub struct QueryMeta {
    /// Total entities returned.
    pub total_entities: usize,
    /// Total edges returned.
    pub total_edges: usize,
    /// Whether there are more results.
    pub has_more: bool,
}

/// Mutation response.
#[derive(Debug, Serialize)]
pub struct MutationResponse {
    /// Success flag.
    pub success: bool,
    /// Number of affected rows.
    pub affected: u64,
    /// Inserted IDs (for insert operations).
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub inserted_ids: Vec<String>,
}

/// Aggregate response.
#[derive(Debug, Serialize)]
pub struct AggregateResponse {
    /// Success flag.
    pub success: bool,
    /// Aggregation results.
    pub data: AggregateDataJson,
}

/// Aggregate data in JSON-friendly format.
#[derive(Debug, Serialize)]
pub struct AggregateDataJson {
    /// Entity type.
    pub entity: String,
    /// Aggregate values.
    pub values: Vec<AggregateValueJson>,
}

/// Single aggregate value in JSON-friendly format.
#[derive(Debug, Serialize)]
pub struct AggregateValueJson {
    /// Function name.
    pub function: String,
    /// Field name (if applicable).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub field: Option<String>,
    /// Computed value.
    pub value: serde_json::Value,
}

/// Replication status response.
#[derive(Debug, Serialize)]
pub struct ReplicationStatusResponse {
    /// Success flag.
    pub success: bool,
    /// Replication status.
    pub data: ReplicationStatusJson,
}

/// Replication status in JSON-friendly format.
#[derive(Debug, Serialize)]
pub struct ReplicationStatusJson {
    /// Role (primary, replica, standalone).
    pub role: String,
    /// Primary address (for replicas).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub primary_addr: Option<String>,
    /// Current LSN.
    pub current_lsn: u64,
    /// Lag in entries (for replicas).
    pub lag_entries: u64,
    /// Lag in milliseconds (for replicas).
    pub lag_ms: u64,
}

/// Stream changes response.
#[derive(Debug, Serialize)]
pub struct StreamChangesResponseJson {
    /// Success flag.
    pub success: bool,
    /// Change entries.
    pub entries: Vec<ChangeLogEntryJson>,
    /// Next LSN for pagination.
    pub next_lsn: u64,
    /// Whether there are more entries.
    pub has_more: bool,
}

/// Change log entry in JSON-friendly format.
#[derive(Debug, Serialize)]
pub struct ChangeLogEntryJson {
    /// LSN.
    pub lsn: u64,
    /// Timestamp (microseconds since epoch).
    pub timestamp: u64,
    /// Entity type.
    pub entity_type: String,
    /// Entity ID (hex).
    pub entity_id: String,
    /// Change type.
    pub change_type: String,
    /// Changed fields.
    pub changed_fields: Vec<String>,
    /// Schema version.
    pub schema_version: u64,
}

// Conversion functions

/// Convert UUID bytes to hex string.
pub fn uuid_to_hex(bytes: &[u8; 16]) -> String {
    bytes.iter().map(|b| format!("{:02x}", b)).collect()
}

/// Convert hex string to UUID bytes.
pub fn hex_to_uuid(hex: &str) -> Result<[u8; 16], String> {
    if hex.len() != 32 {
        return Err("Invalid UUID hex string length".to_string());
    }

    let mut bytes = [0u8; 16];
    for (i, chunk) in hex.as_bytes().chunks(2).enumerate() {
        let hex_str = std::str::from_utf8(chunk).map_err(|e| e.to_string())?;
        bytes[i] = u8::from_str_radix(hex_str, 16).map_err(|e| e.to_string())?;
    }
    Ok(bytes)
}

/// Convert Value to JSON value.
pub fn value_to_json(value: &Value) -> serde_json::Value {
    match value {
        Value::Null => serde_json::Value::Null,
        Value::Bool(b) => serde_json::Value::Bool(*b),
        Value::Int32(i) => serde_json::json!(i),
        Value::Int64(i) => serde_json::json!(i),
        Value::Float32(f) => serde_json::json!(f),
        Value::Float64(f) => serde_json::json!(f),
        Value::String(s) => serde_json::Value::String(s.clone()),
        Value::Bytes(b) => serde_json::Value::String(base64_encode(b)),
        Value::Timestamp(t) => serde_json::json!(t),
        Value::Uuid(u) => serde_json::Value::String(uuid_to_hex(u)),
        Value::BoolArray(arr) => serde_json::json!(arr),
        Value::Int32Array(arr) => serde_json::json!(arr),
        Value::Int64Array(arr) => serde_json::json!(arr),
        Value::Float32Array(arr) => serde_json::json!(arr),
        Value::Float64Array(arr) => serde_json::json!(arr),
        Value::StringArray(arr) => serde_json::json!(arr),
        Value::UuidArray(arr) => {
            let hex_arr: Vec<String> = arr.iter().map(uuid_to_hex).collect();
            serde_json::json!(hex_arr)
        }
        Value::Vector(v) => serde_json::json!(v),
        Value::GeoPoint { lat, lon } => serde_json::json!({ "lat": lat, "lon": lon }),
        Value::GeoPolygon(vertices) => {
            let pts: Vec<serde_json::Value> = vertices
                .iter()
                .map(|(lat, lon)| serde_json::json!({ "lat": lat, "lon": lon }))
                .collect();
            serde_json::json!(pts)
        }
    }
}

/// Base64 encode bytes.
fn base64_encode(data: &[u8]) -> String {
    // Simple base64 encoding without external dependency
    const ALPHABET: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut result = String::with_capacity((data.len() + 2) / 3 * 4);

    for chunk in data.chunks(3) {
        let b0 = chunk[0] as usize;
        let b1 = chunk.get(1).copied().unwrap_or(0) as usize;
        let b2 = chunk.get(2).copied().unwrap_or(0) as usize;

        result.push(ALPHABET[b0 >> 2] as char);
        result.push(ALPHABET[((b0 & 0x03) << 4) | (b1 >> 4)] as char);

        if chunk.len() > 1 {
            result.push(ALPHABET[((b1 & 0x0f) << 2) | (b2 >> 6)] as char);
        } else {
            result.push('=');
        }

        if chunk.len() > 2 {
            result.push(ALPHABET[b2 & 0x3f] as char);
        } else {
            result.push('=');
        }
    }

    result
}

impl From<QueryResult> for QueryResponse {
    fn from(result: QueryResult) -> Self {
        let entities: Vec<EntityBlockJson> = result.entities.iter().map(|block| {
            let rows: Vec<serde_json::Value> = (0..block.ids.len()).map(|i| {
                let mut obj = serde_json::Map::new();
                obj.insert("id".to_string(), serde_json::Value::String(uuid_to_hex(&block.ids[i])));
                for col in &block.columns {
                    obj.insert(col.name.clone(), value_to_json(&col.values[i]));
                }
                serde_json::Value::Object(obj)
            }).collect();

            EntityBlockJson {
                entity: block.entity.clone(),
                rows,
            }
        }).collect();

        let edges: Vec<EdgeBlockJson> = result.edges.iter().map(|block| {
            EdgeBlockJson {
                relation: block.relation.clone(),
                edges: block.edges.iter().map(|e| EdgeJson {
                    from_id: uuid_to_hex(&e.from_id),
                    to_id: uuid_to_hex(&e.to_id),
                }).collect(),
            }
        }).collect();

        let meta = QueryMeta {
            total_entities: result.total_entities(),
            total_edges: result.total_edges(),
            has_more: result.has_more,
        };

        QueryResponse {
            success: true,
            data: QueryResultJson { entities, edges },
            meta,
        }
    }
}

impl From<MutationResult> for MutationResponse {
    fn from(result: MutationResult) -> Self {
        MutationResponse {
            success: true,
            affected: result.affected,
            inserted_ids: result.inserted_ids.iter().map(uuid_to_hex).collect(),
        }
    }
}

impl From<ReplicationStatus> for ReplicationStatusResponse {
    fn from(status: ReplicationStatus) -> Self {
        let (role, primary_addr) = match &status.role {
            ormdb_proto::replication::ReplicationRole::Primary => ("primary".to_string(), None),
            ormdb_proto::replication::ReplicationRole::Replica { primary_addr } => {
                ("replica".to_string(), Some(primary_addr.clone()))
            }
            ormdb_proto::replication::ReplicationRole::Standalone => ("standalone".to_string(), None),
        };

        ReplicationStatusResponse {
            success: true,
            data: ReplicationStatusJson {
                role,
                primary_addr,
                current_lsn: status.current_lsn,
                lag_entries: status.lag_entries,
                lag_ms: status.lag_ms,
            },
        }
    }
}
