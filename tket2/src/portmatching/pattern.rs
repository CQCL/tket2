//! Patterns for hugr pattern matching.
//!
//! We provide two pattern types, with different constraint decomposition
//! logic:
//!
//! -
//!
//! 2. A pattern that matches a single tree hugr.
//!

mod fast;
mod pattern_trait;
mod uf;

pub use fast::{BranchSelectorFast, CircuitPatternFast};
pub use pattern_trait::CircuitPattern;
pub use uf::CircuitPatternUf;

use std::collections::{BTreeMap, BTreeSet};

use derive_more::{Display, Error};
use hugr::{hugr::views::sibling_subgraph::InvalidSubgraph, types::EdgeKind, HugrView};
use itertools::{Either, Itertools};
use portmatching::pattern::ClassRank;
use priority_queue::PriorityQueue;

use super::{
    branch::BranchClass,
    indexing::{HugrNodeID, HugrPath, HugrPortID, HugrVariableID, HugrVariableValue},
    Constraint, Predicate,
};
use crate::{utils::type_is_linear, Circuit};

/// The single source of truth for the VariableMap, mapping incoming ports to paths
type NodeToPathMap = BTreeMap<hugr::Node, HugrPath>;
/// The inverse of `HugrBindMap`
type VariableMap = BTreeMap<HugrVariableValue, HugrVariableID>;

/// Compute canonical names for each node and wire in `circuit`
fn canonical_var_map(circuit: &Circuit<impl HugrView>) -> Result<VariableMap, InvalidPattern> {
    // Find the best map from hugr values to variables
    let path_map = get_node_to_path_map(circuit)?;
    Ok(into_variable_map(path_map, circuit))
}

fn decompose_to_constraints(
    circuit: &Circuit<impl HugrView>,
    var_map: &VariableMap,
) -> BTreeSet<Constraint> {
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
        }

        // Add sink constraints
        for (in_node, in_port) in in_ports {
            if nodes.contains(&in_node) {
                let node_var = var_map[&in_node.into()];
                let args = vec![node_var, wire_id];
                let sink_constraint =
                    Constraint::try_new(Predicate::IsWireSink(in_port), args).unwrap();
                constraints.insert(sink_constraint);
            }
        }
    }

    constraints
}

fn get_io_boundary(
    circuit: &Circuit<impl HugrView>,
    var_map: &VariableMap,
) -> (Vec<HugrPortID>, Vec<HugrPortID>) {
    let nodes: BTreeSet<_> = circuit.commands().map(|cmd| cmd.node()).collect();

    let mut incoming_wires = Vec::new();
    let mut outgoing_wires = Vec::new();

    for (wire, in_ports) in all_circuit_wires(circuit) {
        let wire_id = var_map[&wire.into()];
        let out_node = wire.node();

        if !nodes.contains(&out_node) {
            // This is an input wire
            incoming_wires.push(wire_id.try_into().unwrap());
        }

        if in_ports
            .iter()
            .any(|(in_node, _)| !nodes.contains(&in_node))
        {
            // TODO: for classical wires, it might be useful to register them
            // as outputs even if they are not used as such in the pattern,
            // otherwise there is the risk that in the match these output will
            // be used but won't be available in the boundary.
            outgoing_wires.push(wire_id.try_into().unwrap());
        }
    }

    (incoming_wires, outgoing_wires)
}

fn all_linear_wires(
    circuit: &Circuit<impl HugrView>,
    var_map: &VariableMap,
) -> BTreeSet<HugrPortID> {
    let mut linear_wires = BTreeSet::new();

    for (wire, _) in all_circuit_wires(circuit) {
        let wire_id = var_map[&wire.into()];
        let (out_node, out_port) = (wire.node(), wire.source());

        let is_linear = port_is_linear(out_node, out_port, circuit.hugr());
        if !is_linear {
            continue;
        }

        let HugrVariableID::LinearWire(wire_id) = wire_id else {
            panic!("Invalid key type");
        };
        linear_wires.insert(wire_id);
    }

    linear_wires
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
    let enqueue_neighbours =
        |pq: &mut PriorityQueue<_, _, _>, ret: &BTreeMap<_, _>, node, path: HugrPath| {
            for (traverse_port, node, index) in neighbours(node) {
                let mut path = path.clone();
                // ignore errors, check at the end if all ports have found a path
                if path.push(traverse_port, index).is_ok() && !ret.contains_key(&node) {
                    pq.push_decrease(node, path);
                }
            }
        };

    let mut ret = BTreeMap::new();
    let mut pqueue = PriorityQueue::new();
    pqueue.push(root, HugrPath::empty());

    while let Some((node, new_path)) = pqueue.pop() {
        let success = ret.insert(node, new_path).is_none();
        debug_assert!(success);
        enqueue_neighbours(&mut pqueue, &ret, node, new_path);
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

fn compute_class_rank(cls: BranchClass, n_new_bindings: i32) -> f64 {
    cls.get_rank() * (2_f64.powi(n_new_bindings))
}

/// Find wires that we'd like to check are distinct from all known distinct
/// wires
fn get_distinct_from_classes<'a>(
    known_bindings: &'a [HugrVariableID],
    known_distinct_wires: &'a BTreeSet<HugrPortID>,
    all_wires: impl Iterator<Item = HugrPortID> + 'a,
) -> impl Iterator<Item = (BranchClass, ClassRank)> + 'a {
    // The ports already bound
    let known_ports: BTreeSet<_> = known_bindings
        .iter()
        .filter_map(|&k| HugrPortID::try_from(k).ok())
        .collect();

    all_wires
        .filter(|w| !known_distinct_wires.contains(w))
        .map(move |w| {
            let cls = BranchClass::IsDistinctFromClass(w);

            let args = known_distinct_wires.iter().chain([&w]);
            let n_new_bindings = args.filter(|w| !known_ports.contains(w)).count() as i32;

            let rank = compute_class_rank(cls, n_new_bindings);
            (cls, rank)
        })
}
