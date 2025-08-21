//! Use [`ResourceScope`] to check whether a subcircuit is convex.

use std::collections::{BTreeSet, VecDeque};

use hugr::{Direction, HugrView};

use crate::Subcircuit;

use super::ResourceScope;

impl<H: HugrView> ResourceScope<H> {
    /// Check if the given subcircuit is convex.
    ///
    /// A subcircuit is convex if there is no path from a circuit output to a
    /// circuit input.
    pub fn is_convex(&self, subcircuit: &Subcircuit<H::Node>) -> bool {
        let Some(max_start_pos) = subcircuit
            .intervals_iter()
            .map(|interval| interval.start_pos())
            .max()
        else {
            // An empty subcircuit is convex
            return true;
        };

        let mut future_nodes =
            VecDeque::from_iter(subcircuit.intervals_iter().filter_map(|interval| {
                let last_node = interval.end_node();
                self.resource_path_iter(interval.resource_id(), last_node, Direction::Outgoing)
                    .nth(1)
            }));
        let mut visited = BTreeSet::new();

        // We must prove that all nodes in `future_nodes` are not in the past
        // of any node at the beginning of a line interval.
        while let Some(node) = future_nodes.pop_front() {
            let pos = self.get_position(node).expect("known node");
            if pos > max_start_pos {
                // we cannot be in the past of any node at the beginning of a
                // line interval, so we can stop searching
                continue;
            }
            if !visited.insert(node) {
                continue; // been here before
            }
            for resource_id in self.get_all_resources(node) {
                if let Some(interval) = subcircuit.get_interval(resource_id) {
                    debug_assert!(
                        pos < interval.start_pos() || pos > interval.end_pos(),
                        "node cannot be in interval [min, max]"
                    );
                    if pos < interval.start_pos() {
                        // we are in the past of min, so there is a path from
                        // an output to an input! -> not convex
                        return false;
                    }
                }
            }

            future_nodes.extend(
                self.hugr()
                    .output_neighbours(node)
                    .filter(|&nei| self.contains_node(nei)),
            );
        }

        true
    }
}

#[cfg(test)]
mod tests {
    use crate::{utils::build_simple_circuit, Circuit, TketOp};

    use super::*;

    use rstest::rstest;

    // A circuit made of two CX ladders (v-shape)
    //  - first ladder is (0, 1), (1, 2), etc
    //  - second ladder is (n_qubits - 1, n_qubits - 2), (n_qubits - 2, n_qubits - 3), etc
    fn cx_ladder(n_qubits: usize) -> Circuit {
        build_simple_circuit(n_qubits, |circ| {
            for i in 0..n_qubits - 1 {
                circ.append(TketOp::CX, [i, i + 1]).unwrap();
            }
            for i in (1..n_qubits).rev() {
                circ.append(TketOp::CX, [i, i - 1]).unwrap();
            }
            Ok(())
        })
        .unwrap()
    }

    // Any sequence of non-contiguous node indices will be non-convex.
    // Note that for a lot of non-convex cases, subcircuit construction will
    // fail. We do not include these cases here.
    #[rstest]
    #[case(vec![0, 1], true)]
    #[case(vec![0, 1, 2, 3, 4], true)]
    #[case(vec![4, 5, 6], true)]
    #[case(vec![3, 4], true)]
    #[case(vec![3, 4, 5, 6, 7], true)]
    #[case(vec![0, 2], false)]
    #[case(vec![0, 1, 3, 4], false)]
    #[case(vec![0, 1, 4], false)]
    #[case(vec![3, 6, 7], false)]
    fn test_is_convex(#[case] selected_nodes: Vec<usize>, #[case] is_convex: bool) {
        let circ = cx_ladder(5);
        let subgraph = circ.subgraph();
        let cx_nodes = subgraph.nodes();
        let circ = ResourceScope::from(circ);
        let selected_nodes = selected_nodes.into_iter().map(|i| cx_nodes[i]);

        let subcirc = Subcircuit::try_from_nodes(selected_nodes, &circ).unwrap();

        assert_eq!(circ.is_convex(&subcirc), is_convex);
    }
}
