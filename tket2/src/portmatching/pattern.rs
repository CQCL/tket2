//! Circuit Patterns for pattern matching

mod logic;
mod uf;
use hugr::hugr::views::sibling_subgraph::TopoConvexChecker;
use logic::PatternLogic;
use portmatching::indexing::Binding;
use uf::Uf;

use std::borrow::Borrow;
use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Debug;

use derive_more::{Display, Error};
use hugr::hugr::views::SiblingSubgraph;
use hugr::types::EdgeKind;
use hugr::HugrView;
use itertools::{Either, Itertools};
use portmatching::{self as pm, BindMap};
use priority_queue::PriorityQueue;

use super::indexing::{HugrNodeID, HugrPath, HugrPathBuilder, HugrPortID};
use super::{Constraint, HugrBindMap, HugrVariableID, HugrVariableValue, Predicate};
use crate::rewrite::{InvalidSubgraph, Subcircuit};
use crate::utils::type_is_linear;
use crate::Circuit;

/// The single source of truth for the VariableMap, mapping incoming ports to paths
type NodeToPathMap = BTreeMap<hugr::Node, HugrPath>;
/// The inverse of `HugrBindMap`
type VariableMap = BTreeMap<HugrVariableValue, HugrVariableID>;

/// A pattern that matches a circuit exactly
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct CircuitPattern {
    constraints: BTreeSet<Constraint>,
    incoming_wires: Vec<HugrPortID>,
    outgoing_wires: Vec<HugrPortID>,
    nodes: Vec<HugrNodeID>,
    linear_wires: BTreeSet<HugrPortID>,
}

impl pm::Pattern for CircuitPattern {
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

/// Conversion error from circuit to pattern.
#[derive(Display, Debug, Error, PartialEq, Eq)]
#[non_exhaustive]
pub enum InvalidPattern {
    /// An empty circuit cannot be a pattern.
    #[display("Empty circuits are not allowed as patterns")]
    EmptyCircuit,
    /// Patterns must be connected circuits.
    #[display("The pattern is not connected")]
    NotConnected,
    /// Patterns cannot include empty wires.
    #[display("The pattern contains an empty wire between {from_node}, {from_port} and {to_node}, {to_port}")]
    EmptyWire {
        /// The source node
        from_node: hugr::Node,
        /// The source port
        from_port: hugr::Port,
        /// The target node
        to_node: hugr::Node,
        /// The target port
        to_port: hugr::Port,
    },
    /// A non-linear output port.
    #[display("Found unsupported non-linear pattern output at ({node:?}, {port:?})")]
    NonLinearOutput {
        /// The output node
        node: hugr::Node,
        /// The output port
        port: hugr::OutgoingPort,
    },
}

impl CircuitPattern {
    /// Construct a pattern from a circuit.
    pub fn try_from_circuit<H: HugrView>(circuit: &Circuit<H>) -> Result<Self, InvalidPattern> {
        check_no_empty_wire(circuit)?;

        // Find the best map from hugr values to variables
        let path_map = get_node_to_path_map(circuit)?;
        let var_map = into_variable_map(path_map, circuit);
        let nodes: BTreeSet<_> = circuit.commands().map(|cmd| cmd.node()).collect();

        let mut constraints = BTreeSet::new();

        // 1. Add OpType constraints
        for cmd in circuit.commands() {
            let op = cmd.optype().clone();
            let val = cmd.node().into();
            let var = var_map[&val];
            let pred = Predicate::IsOpEqual(op.into());
            constraints.insert(Constraint::try_new(pred, vec![var]).unwrap());
        }

        let mut linear_wires = BTreeSet::new();
        let mut incoming_wires = Vec::new();
        let mut outgoing_wires = Vec::new();

        // 2. Add wire constraints
        for (wire, in_ports) in all_circuit_wires(circuit) {
            let wire_id = var_map[&wire.into()];
            let (out_node, out_port) = (wire.node(), wire.source());

            // Add source constraint
            if nodes.contains(&out_node) {
                let node_var = var_map[&out_node.into()];
                let args = vec![node_var, wire_id];
                let source_constraint =
                    Constraint::try_new(Predicate::IsWireSource(out_port), args).unwrap();
                constraints.insert(source_constraint);
            } else {
                // This is an input wire
                incoming_wires.push(wire_id.try_into().unwrap());
            }

            // Add sink constraints
            let mut add_output = false;
            for (in_node, in_port) in in_ports {
                if nodes.contains(&in_node) {
                    let node_var = var_map[&in_node.into()];
                    let args = vec![node_var, wire_id];
                    let sink_constraint =
                        Constraint::try_new(Predicate::IsWireSink(in_port), args).unwrap();
                    constraints.insert(sink_constraint);
                } else {
                    // This is an output wire
                    add_output = true;
                }
            }
            if add_output {
                // TODO: for classical wires, it might be useful to register them
                // as outputs even if they are not used as such in the pattern,
                // otherwise there is the risk that in the match these output will
                // be used but won't be available in the boundary.
                outgoing_wires.push(wire_id.try_into().unwrap());
            }

            let is_linear = port_is_linear(out_node, out_port, circuit.hugr());
            if !is_linear {
                continue;
            }

            let HugrVariableID::LinearWire(wire_id) = wire_id else {
                panic!("Invalid key type");
            };
            linear_wires.insert(wire_id);
        }

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
/// Errors that can occur when constructing a circuit match from bindings
#[derive(Debug, Display, Clone, PartialEq, Eq, Error)]
#[non_exhaustive]
pub enum InvalidPatternMatch {
    /// Pattern node cannot be found in bindings
    #[display("pattern node not in match")]
    MissingNodeBinding,

    /// Pattern wire cannot be found in bindings
    #[display("pattern wire not in match")]
    MissingWireBinding,

    /// The matched subcircuit has an unexpected port
    #[display("unexpected port in match")]
    UnexpectedPort,

    /// Matched boundary wire is not within the pattern boundary wires
    #[display("unexpected wire at boundary")]
    UnknownBoundaryWire,

    /// An output is defined more than once in the match
    #[display("more than one source for output")]
    MultipleOutputSources,

    /// Error propagated from subgraph creation: match is not convex, malformed
    /// etc.
    #[display("resulting subgraph is invalid")]
    InvalidSubgraph(InvalidSubgraph),
}

impl From<InvalidSubgraph> for InvalidPatternMatch {
    fn from(e: InvalidSubgraph) -> Self {
        InvalidPatternMatch::InvalidSubgraph(e)
    }
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

fn check_no_empty_wire(circuit: &Circuit<impl HugrView>) -> Result<(), InvalidPattern> {
    let hugr = circuit.hugr();
    let [inp, out] = circuit.io_nodes();

    for out_port in hugr.node_outputs(inp) {
        let mut linked_ports = hugr.linked_inputs(inp, out_port);
        let in_port = linked_ports.find_map(|(n, p)| (n == out).then_some(p));

        if let Some(in_port) = in_port {
            return Err(InvalidPattern::EmptyWire {
                from_node: inp,
                from_port: out_port.into(),
                to_node: out,
                to_port: in_port.into(),
            });
        }
    }
    Ok(())
}

/// Turn a path map for nodes into a map for every variable in `hugr`.
///
/// All non-IO nodes in hugr must be bound in `path_map`.
fn into_variable_map(path_map: NodeToPathMap, circuit: &Circuit<impl HugrView>) -> VariableMap {
    let hugr = circuit.hugr();
    let mut ret = BTreeMap::new();
    let all_nodes = circuit.commands().map(|cmd| cmd.node()).collect_vec();

    // 1. Add nodes
    for &node in &all_nodes {
        let path = path_map[&node];
        let val = HugrVariableValue::Node(node);
        let var = HugrNodeID::new(path);
        ret.insert(val, var.into());
    }

    // 2. Add ports: we identify edges by the unique outgoing port, but we must
    // be careful when this is the input node as it does not have a HugrPathID.
    // In that case we resort to using the first incoming port to identify the
    // wire.
    let out_ports = all_nodes
        .iter()
        .flat_map(|&n| filtered_node_outputs(n, hugr).map(move |p| (n, p)))
        .map(|(n, p)| (n, hugr::Port::from(p)));
    let in_ports = filtered_node_outputs(circuit.input_node(), circuit.hugr())
        .filter_map(|p| circuit.hugr().linked_inputs(circuit.input_node(), p).next())
        .map(|(n, p)| (n, hugr::Port::from(p)));
    let ports = out_ports.chain(in_ports);
    for (node, port) in ports {
        let path = HugrNodeID::new(path_map[&node]);
        let var = HugrPortID::new(path, port.into());
        let (out_node, out_port) = match port.as_directed() {
            Either::Left(in_port) => hugr.single_linked_output(node, in_port).unwrap(),
            Either::Right(out_port) => (node, out_port),
        };
        let val = (out_node, out_port);
        let var = if port_is_linear(node, port, hugr) {
            HugrVariableID::LinearWire(var)
        } else {
            HugrVariableID::CopyableWire(var)
        };
        ret.insert(val.into(), var);
    }

    ret
}

fn port_is_linear(node: hugr::Node, port: impl Into<hugr::Port>, hugr: &impl HugrView) -> bool {
    type_is_linear(
        hugr.signature(node)
            .unwrap()
            .port_type(port.into())
            .unwrap(),
    )
}

/// Fix the incoming port names (i.e. their path)
///
/// Choose the root that minimises the maximum path length.
fn get_node_to_path_map(circuit: &Circuit<impl HugrView>) -> Result<NodeToPathMap, InvalidPattern> {
    if circuit.num_operations() == 0 {
        return Err(InvalidPattern::EmptyCircuit);
    }
    let nodes = circuit.commands().map(|cmd| cmd.node());
    let mut best_map = None;
    let max_path = |map: &BTreeMap<_, _>| map.values().max().copied().unwrap();
    for root in nodes {
        // Use Dijkstra to find all shortest paths from root
        let Some(map) = dijkstra_all_shortest_paths(root, circuit) else {
            continue;
        };
        if best_map.is_none() || max_path(&map) < max_path(best_map.as_ref().unwrap()) {
            best_map = Some(map);
        }
    }
    let Some(best_map) = best_map else {
        return Err(InvalidPattern::NotConnected);
    };
    Ok(best_map)
}

/// Compute the minimum paths from the root to all nodes.
///
/// This is a slightly tweaked version of Dijkstra's algorithm, in which we
/// keep track of the builder object obtained along the shortest path.
///
/// The returned map is a valid [`NodeToPathMap`], i.e. it has a key for every
/// command node in `circuit`.
fn dijkstra_all_shortest_paths(
    root: hugr::Node,
    circuit: &Circuit<impl HugrView>,
) -> Option<NodeToPathMap> {
    let hugr = circuit.hugr();
    let all_nodes: BTreeSet<_> = circuit.commands().map(|cmd| cmd.node()).collect();

    // All neighbours of `node`
    let neighbours = |node| {
        let in_neighbours = filtered_node_inputs(node, hugr).flat_map(|in_port| {
            let (out_node, _) = hugr.single_linked_output(node, in_port)?;
            if !all_nodes.contains(&out_node) {
                return None;
            }
            let traverse = hugr::Port::from(in_port);
            Some((traverse, out_node, 0))
        });
        let out_neighbours = filtered_node_outputs(node, hugr).flat_map(|out_port| {
            let traverse = hugr::Port::from(out_port);
            hugr.linked_inputs(node, out_port)
                .filter(|(n, _)| all_nodes.contains(n))
                .enumerate()
                .map(move |(i, (in_node, _))| (traverse, in_node, i))
        });
        in_neighbours.chain(out_neighbours).collect_vec()
    };
    // Enqueue all unvisited neighbours of `node`.
    let enqueue_neighbours = |pq: &mut PriorityQueue<_, _, _>,
                              ret: &BTreeMap<_, _>,
                              node,
                              path_builder: HugrPathBuilder| {
        for (traverse_port, node, index) in neighbours(node) {
            let mut path_builder = path_builder.clone();
            // ignore errors, check at the end if all ports have found a path
            if path_builder.push(traverse_port, index).is_ok() && !ret.contains_key(&node) {
                pq.push_decrease(node, path_builder);
            }
        }
    };

    let mut ret = BTreeMap::new();
    let mut pqueue = PriorityQueue::new();
    pqueue.push(root, HugrPathBuilder::new());

    while let Some((node, path_builder)) = pqueue.pop() {
        let new_path = path_builder.clone().finish();
        let success = ret.insert(node, new_path).is_none();
        debug_assert!(success);
        enqueue_neighbours(&mut pqueue, &ret, node, path_builder);
    }

    // Check that all nodes have been assigned a path.
    if !all_nodes.iter().all(|node| ret.contains_key(node)) {
        return None;
    }

    Some(ret)
}

/// Iterator over all wires aka values (in a SSA sense) in `circ`.
///
/// For each wire, return the single outgoing port and all incoming ports.
fn all_circuit_wires(
    circ: &Circuit<impl HugrView>,
) -> impl Iterator<Item = (hugr::Wire, Vec<(hugr::Node, hugr::IncomingPort)>)> + '_ {
    let all_outgoing_ports = circ
        .hugr()
        .children(circ.parent())
        .flat_map(|n| filtered_node_outputs(n, circ.hugr()).map(move |p| (n, p)));
    all_outgoing_ports.map(move |(out_node, out_port)| {
        let in_ports = circ.hugr().linked_inputs(out_node, out_port).collect();
        (hugr::Wire::new(out_node, out_port), in_ports)
    })
}

fn filter_ports<'a, P: Into<hugr::Port> + Copy>(
    ports: impl IntoIterator<Item = P> + 'a,
    node: hugr::Node,
    hugr: &'a impl HugrView,
) -> impl Iterator<Item = P> + 'a {
    ports.into_iter().filter(move |&p| {
        let kind = hugr.get_optype(node).port_kind(p);
        matches!(kind, Some(EdgeKind::Const(..) | EdgeKind::Value(..)))
    })
}

fn filtered_node_inputs(
    node: hugr::Node,
    hugr: &impl HugrView,
) -> impl Iterator<Item = hugr::IncomingPort> + '_ {
    let ports = hugr.node_inputs(node);
    filter_ports(ports, node, hugr)
}

fn filtered_node_outputs(
    node: hugr::Node,
    hugr: &impl HugrView,
) -> impl Iterator<Item = hugr::OutgoingPort> + '_ {
    let ports = hugr.node_outputs(node);
    filter_ports(ports, node, hugr)
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

        let p = CircuitPattern::try_from_circuit(&circ).unwrap();

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
            CircuitPattern::try_from_circuit(&circ).unwrap_err(),
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
            CircuitPattern::try_from_circuit(&circ).unwrap_err(),
            InvalidPattern::EmptyWire { .. }
        );
    }

    #[rstest]
    fn pattern_with_copy(circ_with_copy: Circuit) {
        let pattern = CircuitPattern::try_from_circuit(&circ_with_copy).unwrap();

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
            CircuitPattern::try_from_circuit(&circ).unwrap_err(),
            InvalidPattern::NotConnected
        );
    }
}
