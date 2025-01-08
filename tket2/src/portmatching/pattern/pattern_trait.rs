use std::{
    borrow::Borrow,
    collections::{BTreeMap, BTreeSet},
};

use hugr::{
    hugr::views::{
        sibling_subgraph::{InvalidSubgraph, TopoConvexChecker},
        SiblingSubgraph,
    },
    HugrView,
};
use itertools::Itertools;
use portmatching as pm;
use portmatching::{indexing::Binding, BindMap};

use crate::{
    portmatching::{
        indexing::{HugrBindMap, HugrNodeID, HugrPortID, HugrVariableID, HugrVariableValue},
        Constraint, Predicate,
    },
    rewrite::Subcircuit,
    Circuit,
};

use super::{InvalidPattern, InvalidPatternMatch};

/// A pattern to match a circuit.
pub trait CircuitPattern:
    Sized
    + pm::Pattern<Key = HugrVariableID, Constraint = Constraint>
    + Clone
    + serde::Serialize
    + serde::de::DeserializeOwned
{
    /// Construct a pattern from a circuit.
    fn try_from_circuit(circuit: &Circuit<impl HugrView>) -> Result<Self, InvalidPattern>;

    /// The constraints of the pattern
    fn constraints(&self) -> &BTreeSet<Constraint>;

    /// The node IDs of the pattern
    fn nodes(&self) -> &[HugrNodeID];

    /// The incoming boundary wires of the pattern
    fn incoming_wires(&self) -> &[HugrPortID];

    /// The outgoing boundary wires of the pattern
    fn outgoing_wires(&self) -> &[HugrPortID];

    /// The matched subcircuit given a pattern match, using a pre-computed
    /// convexity checker.
    ///
    /// Matches may only require a subset of the avaiable incoming and outgoing
    /// wires. Thus, strictly speaking, may have different boundaries. This is
    /// dealt with at the incoming boundary by leaving empty vectors of incoming
    /// ports, but currently raises an error if it arises at the outgoing
    /// boundary.
    fn get_subcircuit_with_checker<H: HugrView>(
        &self,
        bind_map: &HugrBindMap,
        circuit: &Circuit<H>,
        checker: &TopoConvexChecker<H>,
    ) -> Result<Subcircuit, InvalidPatternMatch> {
        let port_to_wire = get_port_to_wire_map(self.constraints(), bind_map)?;
        let hugr = circuit.hugr();

        // Strategy: find all wires that go from a node in the pattern to a port
        // outside. Then find which wire that corresponds to, and add that to
        // the boundary.

        let nodes = self
            .nodes()
            .iter()
            .map(|&node| {
                get_value(HugrVariableID::Op(node), bind_map)
                    .ok_or(InvalidPatternMatch::MissingNodeBinding)
            })
            .collect::<Result<BTreeSet<_>, _>>()?;

        let mut incoming = vec![vec![]; self.incoming_wires().len()];

        for (node, in_port) in unbound_incoming_wires(&nodes, hugr) {
            let Some(&wire_id) = port_to_wire.get(&(node, in_port.into())) else {
                return Err(InvalidPatternMatch::UnexpectedPort);
            };
            let wire_pos = self
                .incoming_wires()
                .iter()
                .position(|&p| p == wire_id)
                .ok_or(InvalidPatternMatch::UnknownBoundaryWire)?;
            incoming[wire_pos].push((node, in_port));
        }

        let mut outgoing = vec![None; self.outgoing_wires().len()];
        for (node, out_port) in unbound_outgoing_wires(&nodes, hugr) {
            let Some(&wire_id) = port_to_wire.get(&(node, out_port.into())) else {
                return Err(InvalidPatternMatch::UnexpectedPort);
            };
            let wire_pos = self
                .outgoing_wires()
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
    fn get_subcircuit(
        &self,
        bind_map: &HugrBindMap,
        circuit: &Circuit<impl HugrView>,
    ) -> Result<Subcircuit, InvalidPatternMatch> {
        let checker = TopoConvexChecker::new(circuit.hugr());
        self.get_subcircuit_with_checker(bind_map, circuit, &checker)
    }

    /// The number of constraints in the pattern
    fn n_constraints(&self) -> usize {
        self.constraints().len()
    }
}

/// All incoming ports attached to a boundary wire (i.e. a wire whose outgoing
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
