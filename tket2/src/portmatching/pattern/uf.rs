//! Circuit Pattern using Union-Find for pattern matching

mod data_structure;
mod logic;
mod selector;

use data_structure::Uf;
use logic::PatternLogic;

use std::collections::BTreeSet;
use std::fmt::Debug;

use hugr::HugrView;
use itertools::Itertools;
use portmatching as pm;

use super::super::indexing::{HugrNodeID, HugrPortID};
use super::super::{Constraint, HugrVariableID};
use super::{
    all_linear_wires, canonical_var_map, check_no_empty_wire, decompose_to_constraints,
    get_io_boundary, InvalidPattern,
};
use crate::Circuit;

/// A pattern that matches a circuit exactly
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct CircuitPatternUf {
    constraints: BTreeSet<Constraint>,
    incoming_wires: Vec<HugrPortID>,
    outgoing_wires: Vec<HugrPortID>,
    nodes: Vec<HugrNodeID>,
    linear_wires: BTreeSet<HugrPortID>,
}

impl pm::Pattern for CircuitPatternUf {
    type Key = HugrVariableID;
    type Logic = PatternLogic;
    type Constraint = Constraint;

    fn required_bindings(&self) -> Vec<Self::Key> {
        // TODO: We will need the boundary wires too for replacements
        self.nodes.iter().copied().map_into().collect()
    }

    fn into_logic(self) -> Self::Logic {
        PatternLogic::new(self.constraints, self.linear_wires)
    }
}

impl super::CircuitPattern for CircuitPatternUf {
    /// Construct a pattern from a circuit.
    fn try_from_circuit(circuit: &Circuit<impl HugrView>) -> Result<Self, InvalidPattern> {
        check_no_empty_wire(circuit)?;

        // Find the best map from hugr values to variables
        let var_map = canonical_var_map(circuit)?;
        let nodes: BTreeSet<_> = circuit.commands().map(|cmd| cmd.node()).collect();

        let constraints = decompose_to_constraints(circuit, &var_map);
        let (incoming_wires, outgoing_wires) = get_io_boundary(circuit, &var_map);
        let linear_wires = all_linear_wires(circuit, &var_map);

        let nodes = nodes
            .into_iter()
            .map(|n| {
                let HugrVariableID::Op(n) = var_map[&n.into()] else {
                    panic!("Invalid key type");
                };
                n
            })
            .collect();

        Ok(Self {
            constraints,
            incoming_wires,
            outgoing_wires,
            nodes,
            linear_wires,
        })
    }

    fn constraints(&self) -> &BTreeSet<Constraint> {
        &self.constraints
    }

    fn nodes(&self) -> &[HugrNodeID] {
        &self.nodes
    }

    fn incoming_wires(&self) -> &[HugrPortID] {
        &self.incoming_wires
    }

    fn outgoing_wires(&self) -> &[HugrPortID] {
        &self.outgoing_wires
    }
}

#[cfg(test)]
mod tests {

    use cool_asserts::assert_matches;
    use hugr::builder::{DFGBuilder, Dataflow, DataflowHugr};
    use hugr::extension::prelude::qb_t;
    use hugr::types::Signature;
    use rstest::rstest;

    use crate::extension::rotation::rotation_type;
    use crate::portmatching::pattern::CircuitPattern;
    use crate::portmatching::tests::circ_with_copy;
    use crate::portmatching::Predicate;
    use crate::utils::build_simple_circuit;
    use crate::Tk2Op;

    use super::*;

    fn h_cx() -> Circuit {
        build_simple_circuit(2, |circ| {
            circ.append(Tk2Op::CX, [0, 1])?;
            circ.append(Tk2Op::H, [0])?;
            Ok(())
        })
        .unwrap()
    }

    /// A circuit with two rotation gates in parallel, sharing a param
    fn circ_with_copy_disconnected() -> Circuit {
        let input_t = vec![qb_t(), qb_t(), rotation_type()];
        let output_t = vec![qb_t(), qb_t()];
        let mut h = DFGBuilder::new(Signature::new(input_t, output_t)).unwrap();

        let mut inps = h.input_wires();
        let qb1 = inps.next().unwrap();
        let qb2 = inps.next().unwrap();
        let f = inps.next().unwrap();

        let res = h.add_dataflow_op(Tk2Op::Rx, [qb1, f]).unwrap();
        let qb1 = res.outputs().next().unwrap();
        let res = h.add_dataflow_op(Tk2Op::Rx, [qb2, f]).unwrap();
        let qb2 = res.outputs().next().unwrap();

        h.finish_hugr_with_outputs([qb1, qb2]).unwrap().into()
    }

    #[test]
    fn construct_pattern() {
        let circ = h_cx();

        let p = CircuitPatternUf::try_from_circuit(&circ).unwrap();

        insta::assert_debug_snapshot!(p);
    }

    #[test]
    fn disconnected_pattern() {
        let circ = build_simple_circuit(2, |circ| {
            circ.append(Tk2Op::X, [0])?;
            circ.append(Tk2Op::T, [1])?;
            Ok(())
        })
        .unwrap();
        assert_eq!(
            CircuitPatternUf::try_from_circuit(&circ).unwrap_err(),
            InvalidPattern::NotConnected
        );
    }

    #[test]
    fn pattern_with_empty_qubit() {
        let circ = build_simple_circuit(2, |circ| {
            circ.append(Tk2Op::X, [0])?;
            Ok(())
        })
        .unwrap();
        assert_matches!(
            CircuitPatternUf::try_from_circuit(&circ).unwrap_err(),
            InvalidPattern::EmptyWire { .. }
        );
    }

    #[rstest]
    fn pattern_with_copy(circ_with_copy: Circuit) {
        let pattern = CircuitPatternUf::try_from_circuit(&circ_with_copy).unwrap();

        assert_eq!(pattern.incoming_wires.len(), 2);
        assert_eq!(pattern.outgoing_wires.len(), 1);

        let copyable_sink_wires = pattern
            .constraints
            .iter()
            .filter(|c| matches!(c.predicate(), Predicate::IsWireSink(_)))
            .filter_map(|c| {
                let HugrVariableID::CopyableWire(w) = c.required_bindings()[1] else {
                    return None;
                };
                Some(w)
            })
            .collect_vec();

        assert_eq!(copyable_sink_wires.len(), 2);
        assert_eq!(copyable_sink_wires[0], copyable_sink_wires[1]);

        insta::assert_debug_snapshot!(pattern);
    }

    #[test]
    fn pattern_with_copy_disconnected() {
        let circ = circ_with_copy_disconnected();
        assert_eq!(
            CircuitPatternUf::try_from_circuit(&circ).unwrap_err(),
            InvalidPattern::NotConnected
        );
    }
}
