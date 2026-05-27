//! Error types for ormdb-raft.

use thiserror::Error;

/// Errors that can occur in the Raft subsystem.
#[derive(Debug, Error)]
pub enum RaftError {
    /// Error during Raft initialization.
    #[error("Raft initialization failed: {0}")]
    Initialization(String),

    /// Error during storage operations.
    #[error("Storage error: {0}")]
    Storage(String),

    /// Error during network operations.
    #[error("Network error: {0}")]
    Network(String),

    /// Error during serialization/deserialization.
    #[error("Serialization error: {0}")]
    Serialization(String),

    /// Error during write operations.
    #[error("Write error: {0}")]
    Write(String),

    /// Error establishing a linearizable read (ReadIndex / leadership check).
    #[error("Linearizable read error: {0}")]
    Read(String),

    /// Error during membership changes.
    #[error("Membership change error: {0}")]
    MembershipChange(String),

    /// No leader available for write operations.
    #[error("No leader available")]
    NoLeader,

    /// This node is not the leader.
    #[error("Not the leader, current leader is node {leader_id:?} at {leader_addr:?}")]
    NotLeader {
        /// The ID of the current leader.
        leader_id: Option<u64>,
        /// The address of the current leader.
        leader_addr: Option<String>,
    },

    /// Error during snapshot operations.
    #[error("Snapshot error: {0}")]
    Snapshot(String),

    /// Error during shutdown.
    #[error("Shutdown error: {0}")]
    Shutdown(String),

    /// Timeout waiting for operation.
    #[error("Operation timed out: {0}")]
    Timeout(String),

    /// Internal error.
    #[error("Internal error: {0}")]
    Internal(String),
}

impl From<sled::Error> for RaftError {
    fn from(err: sled::Error) -> Self {
        RaftError::Storage(err.to_string())
    }
}

impl From<std::io::Error> for RaftError {
    fn from(err: std::io::Error) -> Self {
        RaftError::Network(err.to_string())
    }
}
