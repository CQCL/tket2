//! Circuit hashing.

use core::panic;
use std::hash::{Hash, Hasher};

use fxhash::{FxHashMap, FxHasher64};
use hugr::hugr::views::HierarchyView;
use hugr::ops::{LeafOp, OpName, OpTag, OpTrait, OpType};
use hugr::types::TypeBound;
use hugr::{HugrView, Node, Port};
use petgraph::visit::{self as pg, Walker};

use super::Circuit;

/// Circuit hashing utilities.
pub trait CircuitHash<'circ>: HugrView {
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
    fn circuit_hash(&'circ self) -> u64;
}

impl<'circ, T> CircuitHash<'circ> for T
where
    T: HugrView + HierarchyView<'circ>,
    for<'a> &'a T:
        pg::GraphBase<NodeId = Node> + pg::IntoNeighborsDirected + pg::IntoNodeIdentifiers,
{
    fn circuit_hash(&'circ self) -> u64 {
        let mut hash_state = HashState::default();

        for node in pg::Topo::new(self).iter(self).filter(|&n| n != self.root()) {
            let hash = hash_node(self, node, &mut hash_state);
            hash_state.set_node(self, node, hash);
        }

        hash_state.get_nonlinear(self.output())
    }
}

/// Auxiliary data for circuit hashing.
///
/// Contains previously computed hashes.
#[derive(Clone, Default, Debug)]
struct HashState {
    /// Computed node hashes for each linear output.
    ///
    /// These are removed from the map when consumed.
    pub linear_hashes: FxHashMap<(Node, Port), u64>,
    /// Computed node hashes.
    ///
    /// Only store hashes for nodes with at least one non-linear output.
    pub nonlinear_hashes: FxHashMap<Node, u64>,
}

impl HashState {
    /// Return the hash for a port.
    ///
    /// If the port was linear, it is removed from the map.
    fn take(&mut self, node: Node, port: Port) -> u64 {
        if let Some(hash) = self.linear_hashes.remove(&(node, port)) {
            return hash;
        }
        self.get_nonlinear(node)
    }

    /// Return the hash for a node.
    fn get_nonlinear(&self, node: Node) -> u64 {
        *self.nonlinear_hashes.get(&node).unwrap()
    }

    /// Register the hash for a node.
    fn set_node(&mut self, circ: &impl HugrView, node: Node, hash: u64) {
        let optype = circ.get_optype(node);
        let signature = optype.signature();
        let mut any_nonlinear = false;
        for (port_type, port) in signature
            .output_types()
            .iter()
            .zip(signature.output_ports())
        {
            match TypeBound::Copyable.contains(port_type.least_upper_bound()) {
                true => any_nonlinear = true,
                false => {
                    self.linear_hashes.insert((node, port), hash);
                }
            }
        }
        if any_nonlinear || optype.tag() <= OpTag::Output {
            self.nonlinear_hashes.insert(node, hash);
        }
    }
}

/// Returns a hashable representation of an operation.
///
/// Panics if the operation is a parametric CustomOp
//
// TODO: Hash custom op parameters. Also the extension name?
fn hashable_op(op: &OpType) -> impl Hash {
    match op {
        OpType::LeafOp(LeafOp::CustomOp(op)) if !op.args().is_empty() => {
            panic!("Parametric operation {} cannot be hashed.", op.name())
        }
        _ => op.name(),
    }
}

/// Compute the hash of a circuit command.
///
/// Uses the hash of the operation and the node hash of its predecessors.
///
/// # Panics
/// - If the command is a container node, or if it is a parametric CustomOp.
/// - If the hash of any of its predecessors has not been set.
fn hash_node(circ: &impl HugrView, node: Node, state: &mut HashState) -> u64 {
    let op = circ.get_optype(node);
    let mut hasher = FxHasher64::default();

    if circ.children(node).count() > 0 {
        panic!("Cannot hash container node {node:?}.");
    }
    hashable_op(op).hash(&mut hasher);

    // Add each each input neighbour hash, including the connected ports.
    // TODO: Ignore state edges?
    for input in circ.node_inputs(node) {
        // Combine the hash for each subport, ignoring their order.
        let input_hash = circ
            .linked_ports(node, input)
            .map(|(pred_node, pred_port)| {
                let pred_node_hash = state.take(pred_node, pred_port);
                fxhash::hash64(&(pred_node_hash, pred_port, input))
            })
            .fold(0, |total, hash| hash ^ total);
        input_hash.hash(&mut hasher);
    }
    hasher.finish()
}

#[cfg(test)]
mod test {
    use hugr::hugr::views::{HierarchyView, SiblingGraph};
    use hugr::HugrView;

    use crate::utils::build_simple_circuit;
    use crate::T2Op;

    use super::*;

    #[test]
    fn hash_equality() {
        let hugr1 = build_simple_circuit(2, |circ| {
            circ.append(T2Op::H, [0])?;
            circ.append(T2Op::T, [1])?;
            circ.append(T2Op::CX, [0, 1])?;
            Ok(())
        })
        .unwrap();
        let circ1: SiblingGraph<'_> = SiblingGraph::new(&hugr1, hugr1.root());
        let hash1 = circ1.circuit_hash();

        // A circuit built in a different order should have the same hash
        let hugr2 = build_simple_circuit(2, |circ| {
            circ.append(T2Op::T, [1])?;
            circ.append(T2Op::H, [0])?;
            circ.append(T2Op::CX, [0, 1])?;
            Ok(())
        })
        .unwrap();
        let circ2: SiblingGraph<'_> = SiblingGraph::new(&hugr2, hugr2.root());
        let hash2 = circ2.circuit_hash();

        assert_eq!(hash1, hash2);

        // Inverting the CX control and target should produce a different hash
        let hugr3 = build_simple_circuit(2, |circ| {
            circ.append(T2Op::T, [1])?;
            circ.append(T2Op::H, [0])?;
            circ.append(T2Op::CX, [1, 0])?;
            Ok(())
        })
        .unwrap();
        let circ3: SiblingGraph<'_> = SiblingGraph::new(&hugr3, hugr3.root());
        let hash3 = circ3.circuit_hash();

        assert_ne!(hash1, hash3);
    }
}
