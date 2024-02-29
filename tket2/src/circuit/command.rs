//! Circuit commands.
//!
//! A [`Command`] is an operation applied to an specific wires, possibly identified by their index in the circuit's input vector.

use std::collections::hash_map::Entry;
use std::collections::{HashMap, HashSet};
use std::iter::FusedIterator;

use hugr::hugr::NodeType;
use hugr::ops::{OpTag, OpTrait};
use hugr::{IncomingPort, OutgoingPort};
use itertools::Either::{self, Left, Right};
use petgraph::visit as pv;

use super::units::{filter, DefaultUnitLabeller, LinearUnit, UnitLabeller, Units};
use super::Circuit;

pub use hugr::ops::OpType;
pub use hugr::types::{EdgeKind, Type, TypeRow};
pub use hugr::{CircuitUnit, Direction, Node, Port, PortIndex, Wire};

/// An operation applied to specific wires.
pub struct Command<'circ, Circ> {
    /// The circuit.
    circ: &'circ Circ,
    /// The operation node.
    node: Node,
    /// An assignment of linear units to the node's ports.
    //
    // We'll need something more complex if `follow_linear_port` stops being a
    // direct map from input to output.
    linear_units: Vec<LinearUnit>,
}

impl<'circ, Circ: Circuit> Command<'circ, Circ> {
    /// Returns the node corresponding to this command.
    #[inline]
    pub fn node(&self) -> Node {
        self.node
    }

    /// Returns the [`NodeType`] of the command.
    #[inline]
    pub fn nodetype(&self) -> &NodeType {
        self.circ.get_nodetype(self.node)
    }

    /// Returns the [`OpType`] of the command.
    #[inline]
    pub fn optype(&self) -> &OpType {
        self.circ.get_optype(self.node)
    }

    /// Returns the units of this command in a given direction.
    #[inline]
    pub fn units(
        &self,
        direction: Direction,
    ) -> impl Iterator<Item = (CircuitUnit, Port, Type)> + '_ {
        match direction {
            Direction::Incoming => Either::Left(self.inputs().map(|(u, p, t)| (u, p.into(), t))),
            Direction::Outgoing => Either::Right(self.outputs().map(|(u, p, t)| (u, p.into(), t))),
        }
    }

    /// Returns the linear units of this command in a given direction.
    #[inline]
    pub fn linear_units(
        &self,
        direction: Direction,
    ) -> impl Iterator<Item = (LinearUnit, Port, Type)> + '_ {
        match direction {
            Direction::Incoming => {
                Either::Left(self.linear_inputs().map(|(u, p, t)| (u, p.into(), t)))
            }
            Direction::Outgoing => {
                Either::Right(self.linear_outputs().map(|(u, p, t)| (u, p.into(), t)))
            }
        }
    }

    /// Returns the linear units of this command in a given direction.
    #[inline]
    pub fn input_qubits(&self) -> impl Iterator<Item = (LinearUnit, IncomingPort, Type)> + '_ {
        self.inputs().filter_map(filter::filter_qubit)
    }

    /// Returns the linear units of this command in a given direction.
    #[inline]
    pub fn output_qubits(&self) -> impl Iterator<Item = (LinearUnit, OutgoingPort, Type)> + '_ {
        self.outputs().filter_map(filter::filter_qubit)
    }

    /// Returns the output units of this command. See [`Command::units`].
    #[inline]
    pub fn outputs(&self) -> Units<OutgoingPort, &'_ Self> {
        Units::new_outgoing(self.circ, self.node, self)
    }

    /// Returns the linear output units of this command. See [`Command::linear_units`].
    #[inline]
    pub fn linear_outputs(&self) -> impl Iterator<Item = (LinearUnit, OutgoingPort, Type)> + '_ {
        self.outputs().filter_map(filter::filter_linear)
    }

    /// Returns the output units and wires of this command.
    #[inline]
    pub fn output_wires(&self) -> impl Iterator<Item = (CircuitUnit, Wire)> + '_ {
        self.outputs().filter_map(move |(unit, port, _typ)| {
            let w = self.assign_wire(self.node, port.into())?;
            Some((unit, w))
        })
    }

    /// Returns the output units of this command.
    #[inline]
    pub fn inputs(&self) -> Units<IncomingPort, &'_ Self> {
        Units::new_incoming(self.circ, self.node, self)
    }

    /// Returns the linear input units of this command. See [`Command::linear_units`].
    #[inline]
    pub fn linear_inputs(&self) -> impl Iterator<Item = (LinearUnit, IncomingPort, Type)> + '_ {
        self.inputs().filter_map(filter::filter_linear)
    }

    /// Returns the input units and wires of this command.
    #[inline]
    pub fn input_wires(&self) -> impl IntoIterator<Item = (CircuitUnit, Wire)> + '_ {
        self.inputs().filter_map(move |(unit, port, _typ)| {
            let w = self.assign_wire(self.node, port.into())?;
            Some((unit, w))
        })
    }

    /// Returns the number of inputs of this command.
    #[inline]
    pub fn input_count(&self) -> usize {
        self.optype().value_input_count() + self.optype().static_input_port().is_some() as usize
    }

    /// Returns the number of outputs of this command.
    #[inline]
    pub fn output_count(&self) -> usize {
        self.optype().value_output_count() + self.optype().static_output_port().is_some() as usize
    }

    /// Returns the port in the command given a linear unit.
    #[inline]
    pub fn linear_unit_port(&self, unit: LinearUnit, direction: Direction) -> Option<Port> {
        self.linear_units(direction)
            .find(|(cu, _, _)| *cu == unit)
            .map(|(_, port, _)| port)
    }

    /// Returns whether the port is a linear port.
    #[inline]
    pub fn is_linear_port(&self, port: Port) -> bool {
        self.optype()
            .port_kind(port)
            .map_or(false, |kind| kind.is_linear())
    }
}

impl<'a, 'circ, Circ: Circuit> UnitLabeller for &'a Command<'circ, Circ> {
    #[inline]
    fn assign_linear(&self, _: Node, port: Port, _linear_count: usize) -> LinearUnit {
        *self.linear_units.get(port.index()).unwrap_or_else(|| {
            panic!(
                "Could not assign a linear unit to port {port:?} of node {:?}",
                self.node
            )
        })
    }

    #[inline]
    fn assign_wire(&self, node: Node, port: Port) -> Option<Wire> {
        match port.as_directed() {
            Left(to_port) => {
                let (from, from_port) = self.circ.linked_outputs(node, to_port).next()?;
                Some(Wire::new(from, from_port))
            }
            Right(from_port) => Some(Wire::new(node, from_port)),
        }
    }
}

impl<'circ, Circ: Circuit> std::fmt::Debug for Command<'circ, Circ> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Command")
            .field("circuit name", &self.circ.name())
            .field("node", &self.node)
            .field("linear_units", &self.linear_units)
            .finish()
    }
}

impl<'circ, Circ> PartialEq for Command<'circ, Circ> {
    fn eq(&self, other: &Self) -> bool {
        self.node == other.node && self.linear_units == other.linear_units
    }
}

impl<'circ, Circ> Eq for Command<'circ, Circ> {}

impl<'circ, Circ> Clone for Command<'circ, Circ> {
    fn clone(&self) -> Self {
        Self {
            circ: self.circ,
            node: self.node,
            linear_units: self.linear_units.clone(),
        }
    }
}

impl<'circ, Circ> std::hash::Hash for Command<'circ, Circ> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.node.hash(state);
        self.linear_units.hash(state);
    }
}

impl<'circ, Circ> PartialOrd for Command<'circ, Circ> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl<'circ, Circ> Ord for Command<'circ, Circ> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.node
            .cmp(&other.node)
            .then(self.linear_units.cmp(&other.linear_units))
    }
}

/// A non-borrowing topological walker over the nodes of a circuit.
type NodeWalker = pv::Topo<Node, HashSet<Node>>;

/// An iterator over the commands of a circuit.
#[derive(Clone)]
pub struct CommandIterator<'circ, Circ> {
    /// The circuit.
    circ: &'circ Circ,
    /// Toposorted nodes.
    nodes: NodeWalker,
    /// Last wire for each [`LinearUnit`] in the circuit.
    wire_unit: HashMap<Wire, usize>,
    /// Remaining commands, not counting I/O nodes.
    remaining: usize,
    /// Delayed output of constant and load const nodes. Contains nodes that
    /// haven't been yielded yet.
    ///
    /// We only yield them as Commands when their consumers require them.
    delayed_consts: HashSet<Node>,
    /// Nodes with delayed predecessors.
    ///
    /// Each node is associated with the number of predecessors that are present
    /// in `delayed_consts`.
    ///
    /// This map is used for performance, to avoid checking the neighbours vs
    /// the `delayed_consts` set for each processed node.
    delayed_consumers: HashMap<Node, usize>,
    /// The next node to be processed.
    ///
    /// This node was produced by the last call to `nodes.next()`, but we had to
    /// yield some delayed const nodes before it.
    delayed_node: Option<Node>,
}

impl<'circ, Circ> CommandIterator<'circ, Circ>
where
    Circ: Circuit,
{
    /// Create a new iterator over the commands of a circuit.
    pub(super) fn new(circ: &'circ Circ) -> Self {
        // Initialize the map assigning linear units to the input's linear
        // ports.
        //
        // TODO: `with_wires` combinator for `Units`?
        let wire_unit = circ
            .linear_units()
            .map(|(linear_unit, port, _)| (Wire::new(circ.input(), port), linear_unit.index()))
            .collect();

        let nodes = pv::Topo::new(&circ.as_petgraph());
        Self {
            circ,
            nodes,
            wire_unit,
            // Ignore the input and output nodes, and the root.
            remaining: circ.node_count() - 3,
            delayed_consts: HashSet::new(),
            delayed_consumers: HashMap::new(),
            delayed_node: None,
        }
    }

    /// Returns the next node to be processed.
    ///
    /// If the next node in the topological order is a constant or load const node,
    /// delay it until its consumers are processed.
    fn next_node(&mut self) -> Option<Node> {
        let node = self
            .delayed_node
            .take()
            .or_else(|| self.nodes.next(&self.circ.as_petgraph()))?;

        // If this node is a constant or load const node, delay it.
        let tag = self.circ.get_optype(node).tag();
        if tag == OpTag::Const || tag == OpTag::LoadConst {
            self.delayed_consts.insert(node);
            for consumer in self.circ.output_neighbours(node) {
                *self.delayed_consumers.entry(consumer).or_default() += 1;
            }
            return self.next_node();
        }

        // Check if we have any delayed const nodes that are consumed by this node.
        match self.delayed_consumers.contains_key(&node) {
            true => {
                let delayed = self.next_delayed_node(node);
                self.delayed_consts.remove(&delayed);
                for consumer in self.circ.output_neighbours(delayed) {
                    let Entry::Occupied(mut entry) = self.delayed_consumers.entry(consumer) else {
                        panic!("Delayed node consumer was not in delayed_consumers. Delayed node: {delayed:?}, consumer: {consumer:?}.");
                    };
                    *entry.get_mut() -= 1;
                    if *entry.get() == 0 {
                        entry.remove();
                    }
                }
                self.delayed_node = Some(node);
                Some(delayed)
            }
            false => Some(node),
        }
    }

    /// Given a node with delayed predecessors, returns one of those predecessors.
    fn next_delayed_node(&mut self, consumer: Node) -> Node {
        let Some(delayed_pred) = self
            .circ
            .input_neighbours(consumer)
            .find(|k| self.delayed_consts.contains(k))
        else {
            panic!("Could not find a delayed predecessor for node {consumer:?}.");
        };

        // Only output this node if it doesn't require any other delayed predecessors.
        match self.delayed_consumers.contains_key(&delayed_pred) {
            true => self.next_delayed_node(delayed_pred),
            false => delayed_pred,
        }
    }

    /// Process a new node, updating wires in `unit_wires`.
    ///
    /// Returns the an option with the `linear_units` used to construct a
    /// [`Command`], if the node is not an input or output.
    ///
    /// We don't return the command directly to avoid lifetime issues due to the
    /// mutable borrow here.
    fn process_node(&mut self, node: Node) -> Option<Vec<LinearUnit>> {
        // The root node is ignored.
        if node == self.circ.root() {
            return None;
        }
        // Inputs and outputs are also ignored.
        // The input wire ids are already set in the `wire_unit` map during initialization.
        let tag = self.circ.get_optype(node).tag();
        if tag == OpTag::Input || tag == OpTag::Output {
            return None;
        }

        // Collect the linear units passing through this command into the map
        // required to construct a `Command`.
        //
        // Updates the map tracking the last wire of linear units.
        let linear_units: Vec<_> = Units::new_outgoing(self.circ, node, DefaultUnitLabeller)
            .filter_map(filter::filter_linear)
            .map(|(_, port, _)| {
                // Find the linear unit id for this port.
                let linear_id = self
                    .follow_linear_port(node, port)
                    .and_then(|input_port| {
                        let input_port = input_port.as_incoming().unwrap();
                        self.circ.linked_outputs(node, input_port).next()
                    })
                    .and_then(|(from, from_port)| {
                        // Remove the old wire from the map (if there was one)
                        self.wire_unit.remove(&Wire::new(from, from_port))
                    })
                    .unwrap_or({
                        // New linear unit found. Assign it a new id.
                        self.wire_unit.len()
                    });
                // Update the map tracking the linear units
                let new_wire = Wire::new(node, port);
                self.wire_unit.insert(new_wire, linear_id);
                LinearUnit::new(linear_id)
            })
            .collect();

        Some(linear_units)
    }

    /// Returns the linear port on the node that corresponds to the same linear unit.
    ///
    /// We assume the linear data uses the same port offsets on both sides of the node.
    /// In the future we may want to have a more general mechanism to handle this.
    //
    // Note that `Command::linear_units` assumes this behaviour.
    fn follow_linear_port(&self, node: Node, port: impl Into<Port>) -> Option<Port> {
        let port = port.into();
        let optype = self.circ.get_optype(node);
        if !optype.port_kind(port)?.is_linear() {
            return None;
        }
        let other_port = Port::new(port.direction().reverse(), port.index());
        if optype.port_kind(other_port) == optype.port_kind(port) {
            Some(other_port)
        } else {
            None
        }
    }
}

impl<'circ, Circ> Iterator for CommandIterator<'circ, Circ>
where
    Circ: Circuit,
{
    type Item = Command<'circ, Circ>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let node = self.next_node()?;
            // Process the node, returning a command if it's not an input or output.
            if let Some(linear_units) = self.process_node(node) {
                self.remaining -= 1;
                return Some(Command {
                    circ: self.circ,
                    node,
                    linear_units,
                });
            }
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }
}

impl<'circ, Circ> FusedIterator for CommandIterator<'circ, Circ> where Circ: Circuit {}

impl<'circ, Circ: Circuit> std::fmt::Debug for CommandIterator<'circ, Circ> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CommandIterator")
            .field("circuit name", &self.circ.name())
            .field("wire_unit", &self.wire_unit)
            .field("remaining", &self.remaining)
            .finish()
    }
}

#[cfg(test)]
mod test {
    use hugr::builder::{Container, DFGBuilder, Dataflow, DataflowHugr};
    use hugr::extension::prelude::QB_T;
    use hugr::ops::handle::NodeHandle;
    use hugr::ops::OpName;
    use hugr::std_extensions::arithmetic::float_ops::FLOAT_OPS_REGISTRY;
    use hugr::std_extensions::arithmetic::float_types::ConstF64;
    use hugr::types::FunctionType;
    use itertools::Itertools;

    use crate::utils::build_simple_circuit;
    use crate::Tk2Op;

    use super::*;

    // We use a macro instead of a function to get the failing line numbers right.
    macro_rules! assert_eq_iter {
        ($iterable:expr, $expected:expr $(,)?) => {
            assert_eq!($iterable.collect_vec(), $expected.into_iter().collect_vec());
        };
    }

    #[test]
    fn iterate_commands() {
        let circ = build_simple_circuit(2, |circ| {
            circ.append(Tk2Op::H, [0])?;
            circ.append(Tk2Op::CX, [0, 1])?;
            circ.append(Tk2Op::T, [1])?;
            Ok(())
        })
        .unwrap();

        assert_eq!(CommandIterator::new(&circ).count(), 3);

        // TODO: Expose the operation names directly in Tk2Op to clean this up
        let tk2op_name = |op: Tk2Op| <Tk2Op as Into<OpType>>::into(op).name();

        let mut commands = CommandIterator::new(&circ);
        assert_eq!(commands.size_hint(), (3, Some(3)));

        let hadamard = commands.next().unwrap();
        assert_eq!(hadamard.optype().name().as_str(), tk2op_name(Tk2Op::H));
        assert_eq_iter!(
            hadamard.inputs().map(|(u, _, _)| u),
            [CircuitUnit::Linear(0)],
        );
        assert_eq_iter!(
            hadamard.outputs().map(|(u, _, _)| u),
            [CircuitUnit::Linear(0)],
        );

        let cx = commands.next().unwrap();
        assert_eq!(cx.optype().name().as_str(), tk2op_name(Tk2Op::CX));
        assert_eq_iter!(
            cx.inputs().map(|(unit, _, _)| unit),
            [CircuitUnit::Linear(0), CircuitUnit::Linear(1)],
        );
        assert_eq_iter!(
            cx.outputs().map(|(unit, _, _)| unit),
            [CircuitUnit::Linear(0), CircuitUnit::Linear(1)],
        );

        let t = commands.next().unwrap();
        assert_eq!(t.optype().name().as_str(), tk2op_name(Tk2Op::T));
        assert_eq_iter!(
            t.inputs().map(|(unit, _, _)| unit),
            [CircuitUnit::Linear(1)],
        );
        assert_eq_iter!(
            t.outputs().map(|(unit, _, _)| unit),
            [CircuitUnit::Linear(1)],
        );

        assert_eq!(commands.next(), None);
    }

    /// Commands iterator with non-linear wires.
    #[test]
    fn commands_nonlinear() {
        let qb_row = vec![QB_T; 1];
        let mut h = DFGBuilder::new(FunctionType::new(qb_row.clone(), qb_row)).unwrap();
        let [q_in] = h.input_wires_arr();

        let constant = h.add_constant(ConstF64::new(0.5)).unwrap();
        let loaded_const = h.load_const(&constant).unwrap();
        let rz = h
            .add_dataflow_op(Tk2Op::RzF64, [q_in, loaded_const])
            .unwrap();

        let circ = h
            .finish_hugr_with_outputs(rz.outputs(), &FLOAT_OPS_REGISTRY)
            .unwrap();

        assert_eq!(CommandIterator::new(&circ).count(), 3);
        let mut commands = CommandIterator::new(&circ);

        // First command is the constant definition.
        // It has a single output.
        let const_cmd = commands.next().unwrap();
        assert_eq!(const_cmd.optype().name().as_str(), "const:custom:f64(0.5)");
        assert_eq_iter!(const_cmd.inputs().map(|(u, _, _)| u), [],);
        assert_eq_iter!(
            const_cmd.outputs().map(|(u, _, _)| u),
            [CircuitUnit::Wire(Wire::new(constant.node(), 0))],
        );

        // Next, the load constant command.
        // It has a single input and a single output.
        let load_const_cmd = commands.next().unwrap();
        let load_const_node = load_const_cmd.node();
        assert_eq!(load_const_cmd.optype().name().as_str(), "LoadConstant");
        assert_eq_iter!(
            load_const_cmd.inputs().map(|(u, _, _)| u),
            [CircuitUnit::Wire(Wire::new(constant.node(), 0))],
        );
        assert_eq_iter!(
            load_const_cmd.outputs().map(|(u, _, _)| u),
            [CircuitUnit::Wire(Wire::new(load_const_node, 0))],
        );

        // Finally, the rz command.
        // It has the qubit and loaded constant as input and a single output.
        let rz_cmd = commands.next().unwrap();
        assert_eq!(rz_cmd.optype().name().as_str(), "quantum.tket2.RzF64");
        assert_eq_iter!(
            rz_cmd.inputs().map(|(u, _, _)| u),
            [
                CircuitUnit::Linear(0),
                CircuitUnit::Wire(Wire::new(load_const_node, 0))
            ],
        );
        assert_eq_iter!(
            rz_cmd.outputs().map(|(u, _, _)| u),
            [CircuitUnit::Linear(0)],
        );
    }
}
