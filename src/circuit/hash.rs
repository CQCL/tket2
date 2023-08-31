//! Circuit hashing.

use core::panic;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};

use hugr::ops::{LeafOp, OpName, OpType};
use hugr::Node;

use super::Circuit;

/// Circuit hashing utilities.
pub trait CircuitHash<'circ>: Circuit<'circ> {
    /// Compute hash of a circuit.
    ///
    /// We compute a hash for each command from its operation and the hash of its
    /// predecessors. The hash of the circuit corresponds to the xor of all its
    /// nodes hashes.
    ///
    /// This hash is independent from the operation traversal order.
    ///
    /// Adapted from Quartz (Apache 2.0)
    /// <https://github.com/quantum-compiler/quartz/blob/2e13eb7ffb3c5c5fe96cf5b4246f4fd7512e111e/src/quartz/tasograph/tasograph.cpp#L410>
    fn circuit_hash(&'circ self) -> u64;
}

impl<'circ, T> CircuitHash<'circ> for T
where
    T: Circuit<'circ>,
{
    fn circuit_hash(&'circ self) -> u64 {
        let mut hash_vals = HashMap::new();

        // FIXME: The name of an operation is not unique enough, as custom ops with
        // different parameters may share a name.
        let hashable_op = |op: &OpType| match op {
            OpType::LeafOp(LeafOp::CustomOp(op)) if !op.args().is_empty() => {
                panic!("Parametric operation {} cannot be hashed.", op.name())
            }
            _ => op.name(),
        };

        // Add a dummy hash for the input node, not exposed by the command iterator.
        hash_vals.insert(self.input(), 0);

        let hash_node = |node, op, hash_vals: &HashMap<Node, u64>| -> u64 {
            let mut hasher = fxhash::FxHasher64::default();
            hashable_op(op).hash(&mut hasher);
            // Add each each input neighbour hash, including the connected ports.
            // TODO: Ignore state edges?
            for input in self.node_inputs(node) {
                // Combine the hash for each subport, ignoring their order.
                let input_hash = self
                    .linked_ports(node, input)
                    .map(|(pred_node, pred_port)| {
                        let pred_node_hash = hash_vals
                            .get(&pred_node)
                            .unwrap_or_else(|| panic!("Missing hash for node {pred_node:?}"));
                        fxhash::hash64(&(pred_node_hash, pred_port, input))
                    })
                    .fold(0, |total, hash| hash ^ total);
                input_hash.hash(&mut hasher);
            }
            hasher.finish()
        };

        for command in self.commands() {
            let node = command.node();
            let op = self.command_optype(&command);
            let hash = hash_node(node, op, &hash_vals);
            hash_vals.insert(command.node(), hash);
        }

        // Add a hash for the output node, based only on it's connections.
        //
        // TODO: The hash of this node should contain all the information of the
        // circuit. We could output this instead of the XOR of all hashes, and
        // avoid storing all of `hash_vals`.
        let hash = hash_node(self.output(), self.get_optype(self.output()), &hash_vals);
        hash_vals.insert(self.output(), hash);

        // Combine all the node hashes in an order-independent operation.
        hash_vals.into_values().fold(0, |total, hash| hash ^ total)
    }
}

#[cfg(test)]
mod test {
    use hugr::hugr::views::{HierarchyView, SiblingGraph};
    use hugr::HugrView;

    use crate::ops::test::build_simple_circuit;
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
