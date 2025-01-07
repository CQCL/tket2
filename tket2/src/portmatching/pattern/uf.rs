//! Circuit Patterns for pattern matching

mod data_structure;
mod logic;
use data_structure::Uf;
use hugr::hugr::views::sibling_subgraph::TopoConvexChecker;
use logic::PatternLogic;
use portmatching::indexing::Binding;

use std::borrow::Borrow;
use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Debug;

use hugr::hugr::views::SiblingSubgraph;
use hugr::HugrView;
use itertools::Itertools;
use portmatching::{self as pm, BindMap};

use super::super::indexing::{HugrNodeID, HugrPortID};
use super::super::{Constraint, HugrBindMap, HugrVariableID, HugrVariableValue, Predicate};
use super::{
    all_linear_wires, canonical_var_map, check_no_empty_wire, decompose_to_constraints,
    get_io_boundary, InvalidPattern, InvalidPatternMatch,
};
use crate::rewrite::{InvalidSubgraph, Subcircuit};
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

impl CircuitPatternUf {
    /// Construct a pattern from a circuit.
    pub fn try_from_circuit<H: HugrView>(circuit: &Circuit<H>) -> Result<Self, InvalidPattern> {
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

    /// The matched subcircuit given a pattern match, using a pre-computed
    /// convexity checker.
    ///
    /// Matches may only require a subset of the avaiable incoming and outgoing
    /// wires. Thus, strictly speaking, may have different boundaries. This is
    /// dealt with at the incoming boundary by leaving empty vectors of incoming
    /// ports, but currently raises an error if it arises at the outgoing
    /// boundary.
    pub fn get_subcircuit_with_checker<H: HugrView>(
        &self,
        bind_map: &HugrBindMap,
        circuit: &Circuit<H>,
        checker: &TopoConvexChecker<H>,
    ) -> Result<Subcircuit, InvalidPatternMatch> {
        let port_to_wire = get_port_to_wire_map(&self.constraints, bind_map)?;
        let hugr = circuit.hugr();

        // Strategy: find all wires that go from a node in the pattern to a port
        // outside. Then find which wire that corresponds to, and add that to
        // the boundary.

        let nodes = self
            .nodes
            .iter()
            .map(|&node| {
                get_value(HugrVariableID::Op(node), bind_map)
                    .ok_or(InvalidPatternMatch::MissingNodeBinding)
            })
            .collect::<Result<BTreeSet<_>, _>>()?;

        let mut incoming = vec![vec![]; self.incoming_wires.len()];

        for (node, in_port) in unbound_incoming_wires(&nodes, hugr) {
            let Some(&wire_id) = port_to_wire.get(&(node, in_port.into())) else {
                return Err(InvalidPatternMatch::UnexpectedPort);
            };
            let wire_pos = self
                .incoming_wires
                .iter()
                .position(|&p| p == wire_id)
                .ok_or(InvalidPatternMatch::UnknownBoundaryWire)?;
            incoming[wire_pos].push((node, in_port));
        }

        let mut outgoing = vec![None; self.outgoing_wires.len()];
        for (node, out_port) in unbound_outgoing_wires(&nodes, hugr) {
            let Some(&wire_id) = port_to_wire.get(&(node, out_port.into())) else {
                return Err(InvalidPatternMatch::UnexpectedPort);
            };
            let wire_pos = self
                .outgoing_wires
                .iter()
                .position(|&p| p == wire_id)
                .ok_or(InvalidPatternMatch::UnknownBoundaryWire)?;
            if outgoing[wire_pos].is_some() {
                return Err(InvalidPatternMatch::MultipleOutputSources);
            }
            outgoing[wire_pos] = Some((node, out_port));
        }

        if outgoing.iter().any(|p| p.is_none()) {
            // TODO: handle discarding classical outputs
            return Err(InvalidPatternMatch::InvalidSubgraph(
                InvalidSubgraph::NotConvex,
            ));
        }
        let outgoing = outgoing.into_iter().map(|p| p.unwrap()).collect_vec();

        let subgraph = SiblingSubgraph::try_new_with_checker(incoming, outgoing, hugr, checker)?;
        Ok(subgraph.into())
    }

    /// The matched subcircuit given a pattern match.
    ///
    /// Matches may only require a subset of the avaiable incoming and outgoing
    /// wires. Thus, strictly speaking, may have different boundaries. This is
    /// dealt with at the incoming boundary by leaving empty vectors of incoming
    /// ports, but currently raises an error if it arises at the outgoing
    /// boundary.
    pub fn get_subcircuit(
        &self,
        bind_map: &HugrBindMap,
        circuit: &Circuit<impl HugrView>,
    ) -> Result<Subcircuit, InvalidPatternMatch> {
        let checker = TopoConvexChecker::new(circuit.hugr());
        self.get_subcircuit_with_checker(bind_map, circuit, &checker)
    }

    /// The number of constraints in the pattern
    pub fn n_constraints(&self) -> usize {
        self.constraints.len()
    }
}

/// All incoming ports attached to a boundary wire (i.e. a wire whose output
/// port is not in the pattern)
fn unbound_incoming_wires<'a>(
    nodes: &'a BTreeSet<hugr::Node>,
    hugr: &'a impl HugrView,
) -> impl Iterator<Item = (hugr::Node, hugr::IncomingPort)> + 'a {
    let all_in_ports = nodes
        .iter()
        .flat_map(|&node| hugr.node_inputs(node).map(move |p| (node, p)));

    all_in_ports.filter(|&(node, in_port)| {
        let out_port = hugr
            .linked_outputs(node, in_port)
            .at_most_one()
            .ok()
            .expect("assuming wires are uniquely identified by output port");
        let Some((out_node, _)) = out_port else {
            return false;
        };
        !nodes.contains(&out_node)
    })
}

/// All outgoing ports attached to a boundary wire (i.e. a wire with at least
/// one input port not in the pattern)
fn unbound_outgoing_wires<'a>(
    nodes: &'a BTreeSet<hugr::Node>,
    hugr: &'a impl HugrView,
) -> impl Iterator<Item = (hugr::Node, hugr::OutgoingPort)> + 'a {
    let all_out_ports = nodes
        .into_iter()
        .flat_map(|&node| hugr.node_outputs(node).map(move |p| (node, p)));

    all_out_ports.filter(|&(node, out_port)| {
        let mut in_ports = hugr.linked_inputs(node, out_port);
        in_ports.any(|(in_node, _)| !nodes.contains(&in_node))
    })
}

/// Map from (node, port) to the ID of the wire in the pattern
type PortWireMap = BTreeMap<(hugr::Node, hugr::Port), HugrPortID>;

fn get_port_to_wire_map(
    constraints: &BTreeSet<Constraint>,
    bind_map: &HugrBindMap,
) -> Result<PortWireMap, InvalidPatternMatch> {
    let all_link_constraints = constraints.iter().filter_map(|c| match *c.predicate() {
        Predicate::IsWireSource(out_port) => Some((out_port.into(), c.required_bindings())),
        Predicate::IsWireSink(in_port) => Some((in_port.into(), c.required_bindings())),
        _ => None,
    });
    let n_p_w_tuples = all_link_constraints.map(|(port, keys)| {
        let node = keys[0];
        let wire = keys[1];
        let node = get_value(node, bind_map).ok_or(InvalidPatternMatch::MissingNodeBinding)?;
        let Ok(wire) = wire.try_into() else {
            return Err(InvalidPatternMatch::MissingWireBinding);
        };
        Ok((node, port, wire))
    });

    let mut map = PortWireMap::new();
    for res in n_p_w_tuples {
        let (n, p, w) = res?;
        map.insert((n, p), w);
    }
    Ok(map)
}

fn get_value<V: TryFrom<HugrVariableValue>>(
    key: HugrVariableID,
    bind_map: &HugrBindMap,
) -> Option<V> {
    let Binding::Bound(val) = bind_map.get_binding(&key) else {
        return None;
    };
    let val = val.borrow().clone();
    val.try_into().ok()
}

#[cfg(test)]
mod tests {

    use cool_asserts::assert_matches;
    use hugr::builder::{DFGBuilder, Dataflow, DataflowHugr};
    use hugr::extension::prelude::qb_t;
    use hugr::types::Signature;
    use rstest::rstest;

    use crate::extension::rotation::rotation_type;
    use crate::portmatching::tests::circ_with_copy;
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
