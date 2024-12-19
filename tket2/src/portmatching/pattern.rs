//! Circuit Patterns for pattern matching

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::fmt::Debug;

use derive_more::{Display, Error};
use hugr::hugr::views::SiblingSubgraph;
use hugr::ops::OpTrait;
use hugr::types::EdgeKind;
use hugr::HugrView;
use itertools::Itertools;
use portmatching::{self as pm, ArityPredicate};
use priority_queue::PriorityQueue;

use super::constraint::{Constraint, Predicate};
use super::indexing::{HugrBindMap, HugrNodeID, HugrPath, HugrPathBuilder, HugrPortID};
use super::{HugrVariableID, HugrVariableValue};
use crate::rewrite::{InvalidSubgraph, Subcircuit};
use crate::Circuit;

/// The single source of truth for the VariableMap, mapping incoming ports to paths
type NodeToPathMap = BTreeMap<hugr::Node, HugrPath>;
/// The inverse of `HugrBindMap`
type VariableMap = BTreeMap<HugrVariableValue, HugrVariableID>;
/// The IO boundary of a pattern
type Boundary = (
    // The input boundary: each inner vec is the list of ports linked to one port at the input node
    Vec<Vec<HugrPortID>>,
    // The output boundary: each port is linked to one port at the output node
    Vec<HugrPortID>,
);

/// A pattern that matches a circuit exactly
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct CircuitPattern {
    constraints: Vec<Constraint>,
    input_ports: Vec<Vec<HugrPortID>>,
    output_ports: Vec<HugrPortID>,
}

impl pm::Pattern for CircuitPattern {
    type Key = HugrVariableID;
    type Predicate = Predicate;
    type Error = ();

    fn try_to_constraint_vec(
        &self,
    ) -> Result<Vec<pm::Constraint<Self::Key, Self::Predicate>>, Self::Error> {
        Ok(self.constraints.clone())
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
        node: hugr::Node,
        port: hugr::OutgoingPort,
    },
}

impl CircuitPattern {
    /// Construct a pattern from a circuit.
    pub fn try_from_circuit<H: HugrView>(circuit: &Circuit<H>) -> Result<Self, InvalidPattern> {
        // Find the best map from hugr values to variables
        let path_map = get_node_to_path_map(circuit)?;
        let var_map = into_variable_map(path_map, circuit);
        let nodes: BTreeSet<_> = circuit.commands().map(|cmd| cmd.node()).collect();

        check_no_empty_wire(circuit)?;

        let mut constraints = Vec::new();

        // 1. Add OpType constraints
        for cmd in circuit.commands() {
            let op = cmd.optype().clone();
            let val = cmd.node().into();
            let var = var_map[&val];
            let pred = Predicate::NodeOp(op.into());
            constraints.push(Constraint::try_new(pred, vec![var]).unwrap());
        }

        // 2. Add Edge constraints
        for (out_port, in_ports) in all_circuit_values(circuit) {
            // Filter out ports that are not in the pattern
            let out_val: Option<HugrVariableValue> = if nodes.contains(&out_port.0) {
                Some(out_port.into())
            } else {
                None
            };
            let in_vals = in_ports
                .into_iter()
                .filter(|(n, _)| nodes.contains(n))
                .map_into()
                .collect_vec();

            // Create the predicate
            let n_in_ports = in_vals.len();
            let has_out_port = out_val.is_some();
            let pred = Predicate::Wire {
                has_out_port,
                n_in_ports,
            };
            if pred.arity() < 2 {
                // This is a tautology
                continue;
            }

            // Create the constraint
            let vars = out_val.into_iter().chain(in_vals).map(|val| var_map[&val]);
            constraints.push(Constraint::try_new(pred, vars.collect()).unwrap());
        }

        // 3. Add NotEqual constraints (for injectivity of pattern match)
        let mut vars = VecDeque::new();
        for node in circuit.commands().map(|cmd| cmd.node()) {
            let var = var_map[&node.into()];
            let n_other = vars.len();
            vars.push_front(var);
            if n_other > 0 {
                let pred = Predicate::IsNotEqual { n_other };
                constraints.push(Constraint::try_new(pred, vars.clone().into()).unwrap());
            }
        }

        // Finally, figure out the input and output ports
        let (input_ports, output_ports) = get_io_boundary(circuit, &var_map)?;

        Ok(Self {
            constraints,
            input_ports,
            output_ports,
        })
    }

    /// The matched subcircuit given a pattern match.
    pub fn get_subcircuit(
        &self,
        bind_map: &HugrBindMap,
        circuit: &Circuit<impl HugrView>,
    ) -> Result<Subcircuit, InvalidSubgraph> {
        let incoming = self
            .input_ports
            .iter()
            .map(|ports| ports.iter().map(|&p| map_port(p, bind_map)).collect_vec())
            .collect();
        let outgoing = self
            .output_ports
            .iter()
            .map(|&p| map_port(p, bind_map))
            .collect_vec();
        let subgraph = SiblingSubgraph::try_new(incoming, outgoing, circuit.hugr())?;
        Ok(subgraph.into())
    }
}

fn get_io_boundary(
    circuit: &Circuit<impl HugrView>,
    var_map: &BTreeMap<HugrVariableValue, HugrVariableID>,
) -> Result<Boundary, InvalidPattern> {
    let [inp, out] = circuit.io_nodes();
    let hugr = circuit.hugr();

    let convert_to_port_var = |val: HugrVariableValue| -> HugrPortID {
        let HugrVariableID::CopyableWire(port) = var_map[&val] else {
            panic!("Invalid variable ID");
        };
        port
    };
    let input_ports = hugr
        .node_outputs(inp)
        .map(|out_port| {
            let in_ports = hugr.linked_inputs(inp, out_port);
            in_ports.map_into().map(convert_to_port_var).collect_vec()
        })
        .collect();
    let output_ports = filtered_node_inputs(out, hugr)
        .map(|in_port| {
            let (out_node, out_port) = hugr
                .single_linked_output(out, in_port)
                .expect("disconnected input port");
            if is_copyable(circuit, out_node, out_port.into()) {
                Err(InvalidPattern::NonLinearOutput {
                    node: out_node,
                    port: out_port,
                })
            } else {
                Ok(convert_to_port_var((out_node, out_port).into()))
            }
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok((input_ports, output_ports))
}

/// Key for sorting constraints.
///
/// Sort by largest variable first, then break ties by predicate.
fn constraint_key(constraint: &Constraint) -> (HugrVariableID, Predicate) {
    let &max_var = constraint.required_bindings().iter().max().unwrap();
    (max_var, constraint.predicate().clone())
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

    // 2. Add ports
    let in_ports = all_nodes
        .iter()
        .flat_map(|&n| filtered_node_inputs(n, hugr).map(move |p| (n, hugr::Port::from(p))));
    let out_ports = all_nodes
        .iter()
        .flat_map(|&n| filtered_node_outputs(n, hugr).map(move |p| (n, hugr::Port::from(p))));
    let ports = in_ports.chain(out_ports);
    for (node, port) in ports {
        let path = HugrNodeID::new(path_map[&node]);
        let var = HugrPortID::new(path, port);
        let val = (node, port);
        ret.insert(val.into(), var.into());
    }

    ret
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
/// node in `hugr`.
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

/// Iterator over all edges aka values (in a SSA sense) in `circ`.
///
/// For each edge, return the ports of the edge on ops of the circuits (i.e. nodes minus input/output).
fn all_circuit_values(
    circ: &Circuit<impl HugrView>,
) -> impl Iterator<
    Item = (
        (hugr::Node, hugr::OutgoingPort),
        Vec<(hugr::Node, hugr::IncomingPort)>,
    ),
> + '_ {
    let all_outgoing_ports = circ
        .hugr()
        .nodes()
        .flat_map(|n| filtered_node_outputs(n, circ.hugr()).map(move |p| (n, p)));
    all_outgoing_ports.map(move |(out_node, out_port)| {
        let in_ports = circ.hugr().linked_inputs(out_node, out_port).collect();
        ((out_node, out_port), in_ports)
    })
}

fn map_port<V: TryFrom<HugrVariableValue>>(port: HugrPortID, bind_map: &HugrBindMap) -> V
where
    V::Error: Debug,
{
    let var = port.into();
    bind_map[&var].clone().try_into().unwrap()
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

fn is_copyable(circuit: &Circuit<impl HugrView>, node: hugr::Node, port: hugr::Port) -> bool {
    circuit
        .hugr()
        .get_optype(node)
        .dataflow_signature()
        .unwrap()
        .port_type(port)
        .unwrap()
        .copyable()
}

#[cfg(test)]
mod tests {

    use std::collections::{HashMap, HashSet};

    use cool_asserts::assert_matches;
    use hugr::builder::{DFGBuilder, Dataflow, DataflowHugr};
    use hugr::extension::prelude::QB_T;
    use hugr::types::Signature;
    use hugr::Direction;
    use rstest::rstest;

    use crate::extension::rotation::ROTATION_TYPE;
    use crate::extension::REGISTRY;
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

        // Assert the optype constraints
        let op_map: HashMap<_, _> = p
            .constraints
            .iter()
            .filter_map(|cons| {
                if let Predicate::NodeOp(op) = cons.predicate() {
                    let HugrVariableID::Op(node) = cons.required_bindings()[0] else {
                        panic!("Invalid variable ID");
                    };
                    Some((op.clone(), node))
                } else {
                    None
                }
            })
            .collect();
        assert_eq!(op_map.len(), 2);
        assert_eq!(op_map[&Tk2Op::H.into()], HugrNodeID::root());
        assert!(matches!(op_map[&Tk2Op::CX.into()], HugrNodeID { .. }));

        // Assert the edge constraints
        let linear_wire_pred = Predicate::Wire {
            has_out_port: true,
            n_in_ports: 1,
        };
        let edge_constraint = p
            .constraints
            .iter()
            .filter(|cons| cons.predicate() == &linear_wire_pred)
            .exactly_one()
            .unwrap();
        let HugrVariableID::CopyableWire(HugrPortID { port, .. }) =
            edge_constraint.required_bindings()[0]
        else {
            panic!("invalid var type");
        };
        assert_eq!(port.direction(), Direction::Outgoing);
        let HugrVariableID::CopyableWire(HugrPortID { port, .. }) =
            edge_constraint.required_bindings()[1]
        else {
            panic!("invalid var type");
        };
        assert_eq!(port.direction(), Direction::Incoming);
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

        // let edges = pattern.pattern.edges().unwrap();
        // let rx_ns = get_nodes_by_tk2op(&circ, Tk2Op::Rx);
        // let inp = circ.input_node();
        // for rx_n in rx_ns {
        //     assert!(edges.iter().any(|e| {
        //         e.reverse().is_none()
        //             && e.source.unwrap() == rx_n.into()
        //             && e.target.unwrap() == NodeID::new_copy(inp, 1)
        //     }));
        // }

        // Assert the optype constraints
        let nodes: HashSet<_> = pattern
            .constraints
            .iter()
            .filter_map(|cons| {
                if let Predicate::NodeOp(op) = cons.predicate() {
                    assert_eq!(op, &Tk2Op::Rx.into());
                    let HugrVariableID::Op(node) = cons.required_bindings()[0] else {
                        panic!("Invalid variable ID");
                    };
                    Some(node)
                } else {
                    None
                }
            })
            .collect();
        assert_eq!(nodes.len(), 2);
        assert!(nodes.contains(&HugrNodeID::root()));

        // Assert the edge constraints
        let constraints = pattern
            .constraints
            .iter()
            .filter(|cons| matches!(cons.predicate(), Predicate::Wire { .. }))
            .collect_vec();

        assert_eq!(constraints.len(), 2);

        // one of the constraints should be a linear wire predicate, with
        // one incoming and one outgoing port
        assert!(constraints.iter().any(|constraint| {
            let has_outgoing = constraint.required_bindings().iter().any(|b| {
                let HugrVariableID::CopyableWire(HugrPortID { port, .. }) = b else {
                    return false;
                };
                port.direction() == Direction::Outgoing
            });
            let has_incoming = constraint.required_bindings().iter().any(|b| {
                let HugrVariableID::CopyableWire(HugrPortID { port, .. }) = b else {
                    return false;
                };
                port.direction() == Direction::Incoming
            });
            let linear_wire_pred = Predicate::Wire {
                has_out_port: true,
                n_in_ports: 1,
            };
            let is_linear = constraint.predicate() == &linear_wire_pred;
            has_outgoing && has_incoming && is_linear
        }));

        // one of the constraints should be a copyable wire predicate and only
        // have incoming ports
        let copyable_wire_pred = Predicate::Wire {
            has_out_port: false,
            n_in_ports: 2,
        };
        assert!(constraints.iter().any(|constraint| {
            let is_copyable = constraint.predicate() == &copyable_wire_pred;
            let all_incoming = constraint.required_bindings().iter().all(|b| {
                let HugrVariableID::CopyableWire(HugrPortID { port, .. }) = b else {
                    return false;
                };
                port.direction() == Direction::Incoming
            });
            is_copyable && all_incoming
        }));
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
