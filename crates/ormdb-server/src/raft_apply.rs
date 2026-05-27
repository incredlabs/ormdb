//! Bridge between Raft consensus and the local storage engine.
//!
//! When ORMDB runs in cluster mode, committed Raft log entries must be applied
//! to each node's local storage. The Raft state machine
//! ([`ormdb_raft::storage::state_machine::OrmdbStateMachine`]) is generic over an
//! application callback ([`ApplyMutationFn`]) to avoid a circular dependency on
//! the server's mutation executor. [`make_apply_fn`] builds that callback from a
//! [`Database`], so a committed `ClientRequest` runs through the same
//! [`MutationExecutor`] as a direct write.
//!
//! This is the apply side of "Raft in the write path": once leader routing
//! submits a mutation via `RaftClusterManager::write`, the entry is replicated,
//! committed, and then applied on every node through this callback.

use std::sync::Arc;

use ormdb_raft::storage::state_machine::ApplyMutationFn;
use ormdb_raft::types::{ClientRequest, ClientResponse};

use crate::database::Database;
use crate::mutation::MutationExecutor;

/// Build the Raft mutation-apply callback for a database.
///
/// The returned closure is invoked by the state machine for every committed,
/// non-noop log entry. It executes the mutation against local storage and maps
/// the result into a [`ClientResponse`].
pub fn make_apply_fn(database: Arc<Database>) -> ApplyMutationFn {
    Arc::new(move |request: &ClientRequest| {
        let executor = MutationExecutor::new(&database);
        match request {
            ClientRequest::Mutate(mutation) => executor
                .execute(mutation)
                .map(ClientResponse::mutation_result)
                .map_err(|e| e.to_string()),
            ClientRequest::MutateBatch(batch) => executor
                .execute_batch(batch)
                .map(ClientResponse::mutation_result)
                .map_err(|e| e.to_string()),
            // Noop is handled by the state machine before reaching the callback.
            ClientRequest::Noop => Ok(ClientResponse::NoopResult),
        }
    })
}
