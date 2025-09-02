//! Applying [`CircuitRewrite`]s to a [`ResourceScope`].

use derive_more::derive::{Display, Error, From};
use hugr::{
    hugr::{
        hugrmut::HugrMut, patch::simple_replace, views::SiblingSubgraph, Patch,
        SimpleReplacementError,
    },
    ops::OpTrait,
    Direction, HugrView, PortIndex,
};
use indexmap::IndexMap;
use itertools::Itertools;

use crate::{
    resource::{
        scope::node_circuit_units_mut, CircuitUnit, Position, ResourceScope, ResourceScopeConfig,
    },
    rewrite::{CircuitRewrite, NewCircuitRewrite, OldCircuitRewrite},
    Circuit,
};

impl<H: HugrMut<Node = hugr::Node>> ResourceScope<H> {
    /// Apply a rewrite to the circuit.
    ///
    /// This should ideally live within [`hugr_core::hugr::Patch`], but would
    /// need ResourceScope: HugrView.
    pub fn apply_rewrite(
        &mut self,
        rewrite: CircuitRewrite<H::Node>,
    ) -> Result<simple_replace::Outcome<H::Node>, CircuitRewriteError> {
        match rewrite {
            CircuitRewrite::New(rewrite) => self.apply_rewrite_new(rewrite),
            CircuitRewrite::Old(OldCircuitRewrite(repl)) => {
                repl.apply(&mut self.hugr).map_err(Into::into)
            }
        }
    }

    fn apply_rewrite_new(
        &mut self,
        rewrite: NewCircuitRewrite<H::Node>,
    ) -> Result<simple_replace::Outcome<H::Node>, CircuitRewriteError> {
        let simple_replacement = rewrite.to_simple_replacement(self);
        let NewCircuitRewrite {
            subcircuit,
            replacement,
        } = rewrite;

        let inputs = subcircuit.input_ports(self);
        let outputs = subcircuit.output_ports(self);

        debug_assert_eq!(inputs.len(), replacement.circuit_signature().input_count());

        let units_at_inputs = inputs.iter().map(|inp| {
            debug_assert!(inp
                .iter()
                .map(|&(node, port)| self.get_circuit_unit(node, port))
                .all_equal());
            debug_assert!(!inp.is_empty());

            let &(node, port) = inp.first().expect("just checked");
            self.get_circuit_unit(node, port).expect("just checked")
        });
        let units_at_outputs = outputs.iter().map(|&(node, port)| {
            self.get_circuit_unit(node, port)
                .expect("output must exist")
        });

        let mut repl_scope = None;
        if replacement.num_operations() > 0 {
            let repl_scope = repl_scope.insert(ResourceScope::from_circuit_with_input_units(
                replacement,
                units_at_inputs,
            ));

            let effective_units_at_outputs =
                repl_scope
                    .subgraph()
                    .outgoing_ports()
                    .iter()
                    .map(|&(node, port)| {
                        repl_scope
                            .get_circuit_unit(node, port)
                            .expect("output must exist")
                    });
            if effective_units_at_outputs
                .zip(units_at_outputs)
                .any(|(u1, u2)| u1 != u2)
            {
                return Err(CircuitRewriteError::ResourcePreservationViolated);
            }

            let max_input_pos = self.get_nearest_position(
                inputs.iter().flatten().map(|&(n, _)| n),
                Direction::Incoming,
            );
            let min_output_pos =
                self.get_nearest_position(outputs.iter().map(|&(n, _)| n), Direction::Outgoing);
            let n_nodes = repl_scope.nodes().len() as i64;
            let ideal_interval_size = Position::new_integer(n_nodes + 2);
            let (start, end) = match (max_input_pos, min_output_pos) {
                (None, None) => (Position::new_integer(0), ideal_interval_size),
                (None, Some(end)) => (Position(end.0 - ideal_interval_size.0), end),
                (Some(start), None) => (start, Position(start.0 + ideal_interval_size.0)),
                (Some(start), Some(end)) => (start, end),
            };
            if start >= end {
                return Err(CircuitRewriteError::EmptyPositionRange);
            }
            repl_scope.rescale_positions(start, end);
        }

        let simple_replace::Outcome {
            node_map,
            removed_nodes,
        } = simple_replacement.apply(&mut self.hugr)?;

        for (&node, _) in &removed_nodes {
            self.circuit_units.swap_remove(&node);
        }

        for (&repl_node, &new_node) in &node_map {
            let repl_scope = repl_scope.as_ref().expect("non-empty replacement");
            if let Some(repl_node_units) = repl_scope.circuit_units.get(&repl_node) {
                self.circuit_units.insert(new_node, repl_node_units.clone());
            }
        }

        let new_node_set = self
            .subgraph()
            .nodes()
            .iter()
            .chain(node_map.values())
            .copied()
            .filter(|n| !removed_nodes.contains_key(&n))
            .collect_vec();

        let [inp_node, out_node] = self.as_circuit().io_nodes();
        let incoming_ports = self
            .hugr()
            .node_outputs(inp_node)
            .map(|p| (inp_node, p))
            .map(|(n, p)| self.hugr().linked_inputs(n, p).collect_vec())
            .take_while(|ports| !ports.is_empty())
            .collect_vec();
        let outgoing_ports = self
            .hugr()
            .node_inputs(out_node)
            .map(|p| (out_node, p))
            .map(|(n, p)| self.hugr().single_linked_output(n, p))
            .while_some()
            .collect_vec();
        self.subgraph =
            SiblingSubgraph::new_unchecked(incoming_ports, outgoing_ports, vec![], new_node_set);

        Ok(simple_replace::Outcome {
            node_map,
            removed_nodes,
        })
    }

    fn get_nearest_position(
        &self,
        nodes: impl IntoIterator<Item = H::Node>,
        dir: Direction,
    ) -> Option<Position> {
        let all_pos = nodes
            .into_iter()
            .flat_map(|n| self.hugr().neighbours(n, dir))
            .filter_map(|n| self.get_position(n));
        match dir {
            Direction::Incoming => all_pos.max(),
            Direction::Outgoing => all_pos.min(),
        }
    }
}

impl<H: Clone + HugrView<Node = hugr::Node>> ResourceScope<H> {
    /// Create a new resource scope from a circuit, using the given input units
    /// instead of allocating new ones.
    fn from_circuit_with_input_units(
        circuit: Circuit<H>,
        units: impl IntoIterator<Item = CircuitUnit<H::Node>>,
    ) -> Self {
        let inputs = circuit
            .circuit_signature()
            .input_ports()
            .map(|p| {
                circuit
                    .hugr()
                    .linked_inputs(circuit.input_node(), p.index())
                    .collect_vec()
            })
            .collect_vec();
        let subgraph = circuit
            .try_to_subgraph()
            .unwrap_or_else(|e| panic!("Invalid subgraph: {e}"));

        let mut this = Self {
            hugr: circuit.into_hugr(),
            subgraph,
            circuit_units: IndexMap::new(),
        };

        for (inp, unit) in inputs.into_iter().zip_eq(units) {
            for (node, port) in inp {
                let node_units = node_circuit_units_mut(&mut this.circuit_units, node, &this.hugr);
                node_units.port_map.set(port, unit)
            }
        }

        let config = ResourceScopeConfig::default();
        this.compute_circuit_units(&config.flows);

        this
    }

    /// Rescale the positions of all nodes to be within the given range.
    fn rescale_positions(&mut self, start: Position, end: Position) {
        let (curr_start, curr_end) = self
            .nodes()
            .iter()
            .filter_map(|&n| (self.get_position(n)))
            .minmax()
            .into_option()
            .expect("non empty subgraph");

        debug_assert!(curr_start < curr_end || (curr_start == curr_end && self.nodes().len() == 1));

        for &node in self.subgraph.nodes() {
            if self.hugr.get_optype(node).dataflow_signature().is_none() {
                continue;
            }
            let node_units = node_circuit_units_mut(&mut self.circuit_units, node, &self.hugr);
            node_units.position = node_units
                .position
                .rescale(curr_start..=curr_end, start..=end);
        }
    }
}

/// Errors that can occur when applying a rewrite to a resource scope.
#[derive(Debug, Display, Error, From)]
pub enum CircuitRewriteError {
    /// An error occurred while applying the rewrite.
    SimpleReplacementError(#[from] SimpleReplacementError),
    /// The replacement could not be inserted in topological order. Is the
    /// subcircuit non-convex or disconnected?
    #[display("replacement could not be inserted in topological order. Is the subcircuit non-convex or disconnected?")]
    EmptyPositionRange,
    /// The rewrite changes the resource paths non-locally.
    #[display("rewrite changes the resource paths non-locally")]
    ResourcePreservationViolated,
}

#[cfg(test)]
mod tests {
    use crate::{
        resource::{CircuitUnit, ResourceScope},
        rewrite::CircuitRewrite,
        utils::build_simple_circuit,
        Subcircuit, TketOp,
    };
    use hugr::{hugr::Patch, ops::OpType, Direction};
    use itertools::Itertools;
    use rstest::rstest;

    fn simple_circuit() -> ResourceScope {
        ResourceScope::from_circuit(
            build_simple_circuit(2, |circ| {
                circ.append(TketOp::H, [0])?;
                circ.append(TketOp::CX, [0, 1])?;
                circ.append(TketOp::CX, [0, 1])?;
                circ.append(TketOp::H, [1])?;
                Ok(())
            })
            .unwrap(),
        )
    }

    fn rewrite_to_n_cx(n: usize) -> (ResourceScope, CircuitRewrite) {
        let circ = simple_circuit();

        let cx_nodes = circ
            .as_circuit()
            .commands()
            .filter_map(|cmd| {
                if &OpType::from(TketOp::CX) == cmd.optype() {
                    Some(cmd.node())
                } else {
                    None
                }
            })
            .collect_array::<2>()
            .unwrap();

        let repl = build_simple_circuit(2, |circ| {
            for _ in 0..n {
                circ.append(TketOp::CX, [0, 1])?;
            }
            Ok(())
        })
        .unwrap();

        let rewrite = CircuitRewrite::try_new(
            Subcircuit::try_from_nodes(cx_nodes, &circ).unwrap(),
            &circ,
            repl,
        )
        .unwrap();

        (circ, rewrite)
    }

    #[rstest]
    #[case(rewrite_to_n_cx(0), "rewrite_to_0cx")]
    #[case(rewrite_to_n_cx(1), "rewrite_to_1cx")]
    #[case(rewrite_to_n_cx(4), "rewrite_to_4cx")]
    fn test_circuit_rewrite_preserves_circuit_units(
        #[case] (circ, rewrite): (ResourceScope, CircuitRewrite),
        #[case] name: &str,
    ) {
        // Apply rewrite to resource scope

        use std::collections::BTreeSet;
        let mut rewritten_scope = circ.clone();
        rewritten_scope.apply_rewrite(rewrite.clone()).unwrap();

        // Apply rewrite to circuit directly
        let mut direct_circuit = circ.as_circuit().extract_dfg().unwrap();
        rewrite
            .to_simple_replacement(&circ)
            .apply(direct_circuit.hugr_mut())
            .unwrap();
        let direct_scope = ResourceScope::from_circuit(direct_circuit);

        assert_eq!(
            BTreeSet::from_iter(rewritten_scope.nodes()),
            BTreeSet::from_iter(direct_scope.nodes())
        );

        // Check that circuit units are identical
        for &node in rewritten_scope.nodes() {
            for dir in Direction::BOTH {
                let rewritten_units = rewritten_scope.get_circuit_units_slice(node, dir);
                let direct_units = direct_scope.get_circuit_units_slice(node, dir);
                assert_eq!(
                    rewritten_units, direct_units,
                    "Circuit units differ for node {:?}",
                    node
                );
            }
        }

        // Check that positions are strictly increasing along each resource path
        for &(node, port) in rewritten_scope.subgraph().incoming_ports().iter().flatten() {
            let res = match rewritten_scope.get_circuit_unit(node, port).unwrap() {
                CircuitUnit::Resource(resource_id) => resource_id,
                CircuitUnit::Copyable(..) => {
                    continue;
                }
            };
            let all_pos = rewritten_scope
                .resource_path_iter(res, node, Direction::Outgoing)
                .map(|n| rewritten_scope.get_position(n).unwrap())
                .collect_vec();
            assert!(all_pos.is_sorted());
        }

        // Snapshot test of all circuit units
        let mut circuit_units = rewritten_scope
            .circuit_units
            .clone()
            .into_iter()
            .collect_vec();
        circuit_units.sort_unstable_by_key(|&(k, _)| k); // For deterministic output

        insta::assert_debug_snapshot!(name, circuit_units);
    }
}
