//! Circuit hashing.

use std::hash::{Hash, Hasher};

use derive_more::{Display, Error};
use fxhash::{FxHashMap, FxHasher64};
use hugr::hugr::views::{HierarchyView, SiblingGraph};
use hugr::ops::{NamedOp, OpType};
use hugr::{HugrView, Node};
use petgraph::visit::{self as pg, Walker};

use super::Circuit;

/// Circuit hashing utilities.
pub trait CircuitHash {
    /// Compute hash of a circuit.
    ///
    /// We compute a hash for each command from its operation and the hash of
    /// its predecessors. The hash of the circuit corresponds to the hash of its
    /// output node.
    ///
    /// This hash is independent from the operation traversal order.
    ///
    /// Adapted from Quartz (Apache 2.0)
    /// <https://github.com/quantum-compiler/quartz/blob/2e13eb7ffb3c5c5fe96cf5b4246f4fd7512e111e/src/quartz/tasograph/tasograph.cpp#L410>
    fn circuit_hash(&self) -> Result<u64, HashError>;
}

impl<T: HugrView> CircuitHash for Circuit<T> {
    fn circuit_hash(&self) -> Result<u64, HashError> {
        let hugr = self.hugr();
        let container: SiblingGraph = SiblingGraph::try_new(hugr, self.parent()).unwrap();
        container.circuit_hash()
    }
}

impl<T> CircuitHash for T
where
    T: HugrView,
{
    fn circuit_hash(&self) -> Result<u64, HashError> {
        let Some([_, output_node]) = self.get_io(self.root()) else {
            return Err(HashError::NotADfg);
        };

        let mut node_hashes = HashState::default();

        for node in pg::Topo::new(&self.as_petgraph())
            .iter(&self.as_petgraph())
            .filter(|&n| n != self.root())
        {
            let hash = hash_node(self, node, &mut node_hashes)?;
            if node_hashes.set_hash(node, hash).is_some() {
                panic!("Hash already set for node {node}");
            }
        }

        // If the output node has no hash, the topological sort failed due to a cycle.
        node_hashes
            .node_hash(output_node)
            .ok_or(HashError::CyclicCircuit)
    }
}

/// Auxiliary data for circuit hashing.
///
/// Contains previously computed hashes.
#[derive(Clone, Default, Debug)]
struct HashState {
    /// Computed node hashes.
    pub hashes: FxHashMap<Node, u64>,
}

impl HashState {
    /// Return the hash for a node.
    #[inline]
    fn node_hash(&self, node: Node) -> Option<u64> {
        self.hashes.get(&node).copied()
    }

    /// Register the hash for a node.
    ///
    /// Returns the previous hash, if it was set.
    #[inline]
    fn set_hash(&mut self, node: Node, hash: u64) -> Option<u64> {
        self.hashes.insert(node, hash)
    }
}

/// Returns a hashable representation of an operation.
fn hashable_op(op: &OpType) -> impl Hash {
    match op {
        OpType::ExtensionOp(op) if !op.args().is_empty() => {
            // TODO: Require hashing for TypeParams?
            format!(
                "{}[{}]",
                op.name(),
                serde_json::to_string(op.args()).unwrap()
            )
        }
        OpType::OpaqueOp(op) if !op.args().is_empty() => {
            format!(
                "{}[{}]",
                op.name(),
                serde_json::to_string(op.args()).unwrap()
            )
        }
        _ => op.name().to_string(),
    }
}

/// Compute the hash of a circuit command.
///
/// Uses the hash of the operation and the node hash of its predecessors.
///
/// # Panics
/// - If the command is a container node, or if it is a parametric CustomOp.
/// - If the hash of any of its predecessors has not been set.
fn hash_node(circ: &impl HugrView, node: Node, state: &mut HashState) -> Result<u64, HashError> {
    let op = circ.get_optype(node);
    let mut hasher = FxHasher64::default();

    // Hash the node children
    if circ.children(node).count() > 0 {
        let container: SiblingGraph = SiblingGraph::try_new(circ, node).unwrap();
        container.circuit_hash()?.hash(&mut hasher);
    }

    // Hash the node operation
    hashable_op(op).hash(&mut hasher);

    // Add each each input neighbour hash, including the connected ports.
    // TODO: Ignore state edges?
    for input in circ.node_inputs(node) {
        // Combine the hash for each subport, ignoring their order.
        let input_hash = circ
            .linked_ports(node, input)
            .map(|(pred_node, pred_port)| {
                let pred_node_hash = state.node_hash(pred_node);
                fxhash::hash64(&(pred_node_hash, pred_port, input))
            })
            .fold(0, |total, hash| hash ^ total);
        input_hash.hash(&mut hasher);
    }
    Ok(hasher.finish())
}

/// Errors that can occur while hashing a hugr.
#[derive(Debug, Display, Clone, PartialEq, Eq, Error)]
#[non_exhaustive]
pub enum HashError {
    /// The circuit contains a cycle.
    #[display("The circuit contains a cycle.")]
    CyclicCircuit,
    /// The hashed hugr is not a DFG.
    #[display("Tried to hash a non-dfg hugr.")]
    NotADfg,
}

#[cfg(test)]
mod test {
    use tket_json_rs::circuit_json;

    use crate::serialize::TKETDecode;
    use crate::utils::build_simple_circuit;
    use crate::Tk2Op;

    use super::*;

    #[test]
    fn hash_equality() {
        let circ1 = build_simple_circuit(2, |circ| {
            circ.append(Tk2Op::H, [0])?;
            circ.append(Tk2Op::T, [1])?;
            circ.append(Tk2Op::CX, [0, 1])?;
            Ok(())
        })
        .unwrap();
        let hash1 = circ1.circuit_hash().unwrap();

        // A circuit built in a different order should have the same hash
        let circ2 = build_simple_circuit(2, |circ| {
            circ.append(Tk2Op::T, [1])?;
            circ.append(Tk2Op::H, [0])?;
            circ.append(Tk2Op::CX, [0, 1])?;
            Ok(())
        })
        .unwrap();
        let hash2 = circ2.circuit_hash().unwrap();

        assert_eq!(hash1, hash2);

        // Inverting the CX control and target should produce a different hash
        let circ3 = build_simple_circuit(2, |circ| {
            circ.append(Tk2Op::T, [1])?;
            circ.append(Tk2Op::H, [0])?;
            circ.append(Tk2Op::CX, [1, 0])?;
            Ok(())
        })
        .unwrap();
        let hash3 = circ3.circuit_hash().unwrap();

        assert_ne!(hash1, hash3);
    }

    #[test]
    fn hash_constants() {
        let c_str = r#"{"bits": [], "commands": [{"args": [["q", [0]]], "op": {"params": ["0.5"], "type": "Rz"}}], "created_qubits": [], "discarded_qubits": [], "implicit_permutation": [[["q", [0]], ["q", [0]]]], "phase": "0.0", "qubits": [["q", [0]]]}"#;
        let ser: circuit_json::SerialCircuit = serde_json::from_str(c_str).unwrap();
        let circ: Circuit = ser.decode().unwrap();
        circ.circuit_hash().unwrap();
    }

    #[test]
    fn hash_constants_neq() {
        let c_str1 = r#"{"bits": [], "commands": [{"args": [["q", [0]]], "op": {"params": ["0.5"], "type": "Rz"}}], "created_qubits": [], "discarded_qubits": [], "implicit_permutation": [[["q", [0]], ["q", [0]]]], "phase": "0.0", "qubits": [["q", [0]]]}"#;
        let c_str2 = r#"{"bits": [], "commands": [{"args": [["q", [0]]], "op": {"params": ["1.0"], "type": "Rz"}}], "created_qubits": [], "discarded_qubits": [], "implicit_permutation": [[["q", [0]], ["q", [0]]]], "phase": "0.0", "qubits": [["q", [0]]]}"#;

        let mut all_hashes = Vec::with_capacity(2);
        for c_str in [c_str1, c_str2] {
            let ser: circuit_json::SerialCircuit = serde_json::from_str(c_str).unwrap();
            let circ: Circuit = ser.decode().unwrap();
            all_hashes.push(circ.circuit_hash().unwrap());
        }
        assert_ne!(all_hashes[0], all_hashes[1]);
    }
}
