//! Circuit Patterns for pattern matching

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::fmt::Debug;

use derive_more::{Display, Error};
use hugr::hugr::views::SiblingSubgraph;
use hugr::types::EdgeKind;
use hugr::HugrView;
use itertools::{Either, Itertools};
use portmatching as pm;
use priority_queue::PriorityQueue;

use super::constraint::{Constraint, Predicate};
use super::indexing::{HugrBindMap, HugrNodeID, HugrPath, HugrPathBuilder, HugrPortID};
use super::{HugrVariableID, HugrVariableValue};
use crate::rewrite::{InvalidSubgraph, Subcircuit};
use crate::Circuit;

/// The single source of truth for the VariableMap, mapping incoming ports to paths
type InPortToPathMap = BTreeMap<(hugr::Node, hugr::IncomingPort), HugrPath>;
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
    type Constraint = Constraint;

    type Error = ();

    fn try_to_constraint_vec(&self) -> Result<Vec<Self::Constraint>, Self::Error> {
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
}

impl CircuitPattern {
    /// Construct a pattern from a circuit.
    pub fn try_from_circuit<H: HugrView>(circuit: &Circuit<H>) -> Result<Self, InvalidPattern> {
        // Find the best map from hugr values to variables
        let (root, port_map) = get_in_port_to_path_map(circuit)?;
        let var_map = into_variable_map(port_map, root, circuit)?;

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
        for ports in all_circuit_values(circuit) {
            for (&(node1, port1), &(node2, port2)) in ports.iter().tuple_windows() {
                let vals = [(node1, port1).into(), (node2, port2).into()];
                let vars = vals.map(|val| var_map[&val]);
                let pred = Predicate::ShareEdge(port1, port2);
                constraints.push(Constraint::try_new(pred, vars.to_vec()).unwrap());
            }
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
        let (input_ports, output_ports) = get_io_boundary(circuit, &var_map);

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
) -> Boundary {
    let [inp, out] = circuit.io_nodes();
    let hugr = circuit.hugr();

    let convert_to_port_var = |val: HugrVariableValue| -> HugrPortID {
        let HugrVariableID::Port(port) = var_map[&val] else {
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
            let out_port = hugr
                .single_linked_output(out, in_port)
                .expect("disconnected input port");
            convert_to_port_var(out_port.into())
        })
        .collect();
    (input_ports, output_ports)
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

/// Turn a path map for incoming ports into a map of every variable in `hugr`.
fn into_variable_map(
    port_map: InPortToPathMap,
    root: hugr::Node,
    circuit: &Circuit<impl HugrView>,
) -> Result<VariableMap, InvalidPattern> {
    let hugr = circuit.hugr();
    let mut ret = BTreeMap::new();

    // 1. Add incoming ports
    let all_incoming_ports = hugr
        .nodes()
        .flat_map(|n| filtered_node_inputs(n, hugr).map(move |p| (n, p)));
    for (node, port) in all_incoming_ports {
        let path = port_map[&(node, port)];
        let var = HugrPortID::new_incoming(path).into();
        let val = HugrVariableValue::IncomingPort(node, port);
        ret.insert(val, var);
    }

    // 2. Add outgoing ports
    let all_outgoing_ports = hugr
        .nodes()
        .flat_map(|n| hugr.node_outputs(n).map(move |p| (n, p)));
    for (node, port) in all_outgoing_ports {
        let Some((node, in_port)) = hugr.linked_inputs(node, port).next() else {
            // Ignore ports that are not connected to an input
            continue;
        };
        let opposite_port = port_map[&(node, in_port)];
        let var = HugrPortID::new_outgoing(opposite_port).into();
        let val = HugrVariableValue::OutgoingPort(node, port);
        ret.insert(val, var);
    }

    // 3. Add nodes
    let all_nodes = circuit.commands().map(|cmd| cmd.node());
    for node in all_nodes {
        let val = HugrVariableValue::Node(node);
        let var = get_node_var(node, root, hugr, &port_map)?;
        ret.insert(val, var.into());
    }
    Ok(ret)
}

/// Get the variable ID for a node.
fn get_node_var(
    node: hugr::Node,
    root: hugr::Node,
    hugr: &impl HugrView,
    port_map: &InPortToPathMap,
) -> Result<HugrNodeID, InvalidPattern> {
    if node == root {
        Ok(HugrNodeID::Root)
    } else {
        let in_ports = filtered_node_inputs(node, hugr).map_into();
        let connected_out_ports = hugr
            .node_outputs(node)
            .filter(|&p| hugr.is_linked(node, p))
            .map_into();
        // Any incident port (there must be one as node is not root)
        let incident_port: hugr::Port = in_ports
            .chain(connected_out_ports)
            .next()
            .ok_or(InvalidPattern::NotConnected)?;
        // Map port to a variable ID
        let incident_port = match incident_port.as_directed() {
            Either::Left(in_port) => {
                let path = port_map[&(node, in_port)];
                HugrPortID::new_incoming(path)
            }
            Either::Right(out_port) => {
                let in_port = hugr
                    .linked_inputs(node, out_port)
                    .next()
                    .expect("checked above: port is connected");
                let path = port_map[&in_port];
                HugrPortID::new_outgoing(path)
            }
        };
        Ok(HugrNodeID::NonRoot { incident_port })
    }
}

/// Fix the incoming port names (i.e. their path)
///
/// Choose the root that minimises the maximum path length.
fn get_in_port_to_path_map(
    circuit: &Circuit<impl HugrView>,
) -> Result<(hugr::Node, InPortToPathMap), InvalidPattern> {
    if circuit.num_operations() == 0 {
        return Err(InvalidPattern::EmptyCircuit);
    }
    let nodes = circuit.commands().map(|cmd| cmd.node());
    let mut best_map = None;
    let mut best_root = None;
    let max_path = |map: &BTreeMap<_, _>| {
        map.values()
            .map(|path: &HugrPath| -> usize { (*path).into() })
            .max()
            .unwrap()
    };
    for root in nodes {
        // Use Dijkstra to find all shortest paths from root
        let Some(map) = dijkstra_all_shortest_paths(root, circuit.hugr()) else {
            continue;
        };
        if best_map.is_none() || max_path(&map) < max_path(best_map.as_ref().unwrap()) {
            best_map = Some(map);
            best_root = Some(root);
        }
    }
    let Some(best_root) = best_root else {
        return Err(InvalidPattern::NotConnected);
    };
    let best_map = best_map.unwrap();
    Ok((best_root, best_map))
}

/// Compute the minimum paths from the root to all nodes.
///
/// The returned map is a valid InPortToPathMap, i.e. it has a key for every
/// input port in `hugr`.
fn dijkstra_all_shortest_paths(root: hugr::Node, hugr: &impl HugrView) -> Option<InPortToPathMap> {
    // Iterate over all incoming ports i) at `node` and ii) at successors of `node`.
    let iter_incoming = |node| {
        let self_in_ports = filtered_node_inputs(node, hugr).map(move |p| (node, p, 0));
        let next_in_ports = hugr
            .node_outputs(node)
            .enumerate()
            .flat_map(move |(i, p)| hugr.linked_inputs(node, p).map(move |(n, p)| (n, p, i)));
        self_in_ports.chain(next_in_ports)
    };
    // Enqueue all ports for `node`.
    let enqueue = |pq: &mut PriorityQueue<_, _, _>,
                   ret: &BTreeMap<_, _>,
                   node,
                   path_builder: HugrPathBuilder| {
        for (node, port, index) in iter_incoming(node) {
            let mut path_builder = path_builder.clone();
            // ignore errors, check at the end if all ports have found a path
            if path_builder.push(port.into(), index).is_ok() && !ret.contains_key(&(node, port)) {
                pq.push_decrease((node, port), path_builder);
            }
        }
    };

    let mut ret = BTreeMap::new();
    let mut pqueue = PriorityQueue::new();
    enqueue(&mut pqueue, &ret, root, HugrPathBuilder::new());

    while let Some(((node, port), path_builder)) = pqueue.pop() {
        let new_path = path_builder.clone().finish();
        let success = ret.insert((node, port), new_path).is_none();
        debug_assert!(success);
        enqueue(&mut pqueue, &ret, node, path_builder);
    }

    // Check that all input ports have been assigned a path.
    let all_in_ports = hugr
        .nodes()
        .flat_map(|n| filtered_node_inputs(n, hugr).map(move |p| (n, p)));
    for (node, port) in all_in_ports {
        if !ret.contains_key(&(node, port)) {
            return None;
        }
    }

    Some(ret)
}

/// Iterator over all edges aka values (in a SSA sense) in `circ`.
///
/// For each edge, return the ports of the edge on ops of the circuits (i.e. nodes minus input/output).
fn all_circuit_values(
    circ: &Circuit<impl HugrView>,
) -> impl Iterator<Item = Vec<(hugr::Node, hugr::Port)>> + '_ {
    let all_outgoing_ports = circ
        .hugr()
        .nodes()
        .flat_map(|n| circ.hugr().node_outputs(n).map(move |p| (n, p)));
    let op_nodes: BTreeSet<_> = circ.commands().map(|cmd| cmd.node()).collect();
    // this assumes that there is a one-to-many mapping between outgoing ports and incoming ports (i.e. copy fan-out)
    all_outgoing_ports.map(move |(out_node, out_port)| {
        let in_ports = circ
            .hugr()
            .linked_inputs(out_node, out_port)
            .map(|(n, p)| (n, p.into()));
        let ports = in_ports
            .chain([(out_node, out_port.into())])
            .filter(|(n, _)| op_nodes.contains(n));
        ports.collect()
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

#[cfg(test)]
mod tests {

    use std::collections::{HashMap, HashSet};

    use cool_asserts::assert_matches;
    use hugr::builder::{DFGBuilder, Dataflow, DataflowHugr};
    use hugr::extension::prelude::QB_T;
    use hugr::types::Signature;
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
        let input_t = vec![QB_T, QB_T, ROTATION_TYPE];
        let output_t = vec![QB_T, QB_T];
        let mut h = DFGBuilder::new(Signature::new(input_t, output_t)).unwrap();

        let mut inps = h.input_wires();
        let qb1 = inps.next().unwrap();
        let qb2 = inps.next().unwrap();
        let f = inps.next().unwrap();

        let res = h.add_dataflow_op(Tk2Op::Rx, [qb1, f]).unwrap();
        let qb1 = res.outputs().next().unwrap();
        let res = h.add_dataflow_op(Tk2Op::Rx, [qb2, f]).unwrap();
        let qb2 = res.outputs().next().unwrap();

        h.finish_hugr_with_outputs([qb1, qb2], &REGISTRY)
            .unwrap()
            .into()
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
                    let HugrVariableID::Node(node) = cons.required_bindings()[0] else {
                        panic!("Invalid variable ID");
                    };
                    Some((op.clone(), node))
                } else {
                    None
                }
            })
            .collect();
        assert_eq!(op_map.len(), 2);
        assert_eq!(op_map[&Tk2Op::CX.into()], HugrNodeID::Root);
        assert!(matches!(
            op_map[&Tk2Op::H.into()],
            HugrNodeID::NonRoot { .. }
        ));

        // Assert the edge constraints
        let edge_constraint = p
            .constraints
            .iter()
            .filter(|cons| matches!(cons.predicate(), &Predicate::ShareEdge { .. }))
            .exactly_one()
            .unwrap();
        assert!(matches!(
            edge_constraint.predicate(),
            Predicate::ShareEdge { .. }
        ));
        assert!(matches!(
            edge_constraint.required_bindings()[0],
            HugrVariableID::Port(HugrPortID::Incoming { .. })
        ));
        assert!(matches!(
            edge_constraint.required_bindings()[1],
            HugrVariableID::Port(HugrPortID::Outgoing { .. })
        ));
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
                    let HugrVariableID::Node(node) = cons.required_bindings()[0] else {
                        panic!("Invalid variable ID");
                    };
                    Some(node)
                } else {
                    None
                }
            })
            .collect();
        assert_eq!(nodes.len(), 2);
        assert!(nodes.contains(&HugrNodeID::Root));

        // Assert the edge constraints
        let (c1, c2) = pattern
            .constraints
            .iter()
            .filter(|cons| matches!(cons.predicate(), &Predicate::ShareEdge { .. }))
            .collect_tuple()
            .expect("expected two constraints");
        assert!(matches!(c1.predicate(), Predicate::ShareEdge { .. }));
        assert!(matches!(c2.predicate(), Predicate::ShareEdge { .. }));
        // one of c1, c2 should have only incoming ports
        assert!([c1, c2].iter().any(|constraint| {
            matches!(
                constraint.required_bindings()[0],
                HugrVariableID::Port(HugrPortID::Incoming { .. })
            ) || matches!(
                constraint.required_bindings()[1],
                HugrVariableID::Port(HugrPortID::Incoming { .. })
            )
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
