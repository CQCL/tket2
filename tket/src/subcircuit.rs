//! Subcircuits of circuits.
//!
//! Subcircuits are subgraphs of [`hugr::Hugr`] that use a pre-computed
//! [`ResourceScope`] to express subgraphs in terms of intervals on resource
//! paths.

use std::collections::BTreeMap;

use derive_more::derive::{Display, Error};
use hugr::core::HugrNode;
use hugr::hugr::views::sibling_subgraph::{InvalidReplacement, InvalidSubgraph};
use hugr::hugr::views::SiblingSubgraph;
use hugr::{Direction, HugrView, Wire};
use itertools::Itertools;

use crate::circuit::Circuit;
use crate::resource::{Interval, InvalidInterval, ResourceId, ResourceScope};
use crate::rewrite::CircuitRewrite;

/// A subgraph within a [`ResourceScope`].
///
/// Store subgraphs of [`ResourceScope`]s as intervals on the resource paths of
/// the circuit (see below and [`ResourceScope`]). Convex subgraphs can always
/// be represented by intervals; some non-convex subgraphs can also be
/// expressed, as long as for each resource path within the subgraph, the nodes
/// on that path are connected.
///
/// ## Differences with [`SiblingSubgraph`]
///
/// There are some performance and feature distinctions between [`Subcircuit`]s
/// and [`SiblingSubgraph`] to be aware of:
///  - Subcircuit are typically more memory efficient. The size of a
///    [`Subcircuit`] instance is not linear in the number of nodes in the
///    subgraph, but linear in the number of resources (~qubits) in the
///    subcircuit.
///  - Subcircuits may store some non-convex subgraphs, so conversion to
///    [`SiblingSubgraph`] may fail.
///  - Subcircuits can be updated by extending the intervals without having to
///    recompute the subgraph from scratch or rechecking convexity.
///  - Subcircuits currently do not support ops with copyable outputs. In
///    particular, all copyable values in the subcircuit will always be inputs.
#[derive(Debug, Clone, PartialEq)]
pub struct Subcircuit<N: HugrNode = hugr::Node> {
    intervals: Vec<Interval<N>>,
    /// The subcircuit inputs that are linear values
    input_resources: Vec<ResourceId>,
    /// The subcircuit inputs that are copyable values
    input_copyable_values: Vec<Wire<N>>,
    /// The subcircuit outputs (only linear values supported)
    output_resources: Vec<ResourceId>,
}

impl<N: HugrNode> Default for Subcircuit<N> {
    fn default() -> Self {
        Self {
            intervals: Vec::new(),
            input_resources: Vec::new(),
            input_copyable_values: Vec::new(),
            output_resources: Vec::new(),
        }
    }
}

/// Errors that can occur when creating a [`Subcircuit`].
#[derive(Debug, Clone, PartialEq, Display, Error)]
pub enum InvalidSubcircuit<N> {
    /// Copyable values at the output are currently not supported.
    #[display("unsupported copyable output values in {_0:?}")]
    OutputCopyableValues(N),
    /// The node is not contiguous with the subcircuit.
    #[display("node {_0:?} is not contiguous with the subcircuit")]
    NotContiguous(N),
}

impl<N: HugrNode> Subcircuit<N> {
    /// Create a new subcircuit induced from a single node.
    #[inline(always)]
    pub fn from_node(node: N, circuit: &ResourceScope<impl HugrView<Node = N>>) -> Self {
        Self::try_from_nodes([node], circuit).expect("single node is a valid subcircuit")
    }

    /// Create a new subcircuit induced from a set of nodes.
    pub fn try_from_nodes(
        nodes: impl IntoIterator<Item = N>,
        circuit: &ResourceScope<impl HugrView<Node = N>>,
    ) -> Result<Self, InvalidSubcircuit<N>> {
        // For each resource, track the largest interval that contains all nodes,
        // as well as the number of nodes in the interval.
        let mut intervals: BTreeMap<ResourceId, (Interval<N>, usize)> = BTreeMap::new();
        let mut input_copyable_values = Vec::new();

        for node in nodes {
            extend_intervals(&mut intervals, node, circuit);
            update_copyable_inputs(&mut input_copyable_values, node, circuit)?;
        }

        // Check that all intervals are full, i.e. all expected nodes are present
        for &(interval, num_nodes) in intervals.values() {
            let exp_num_nodes = circuit.nodes_in_interval(interval).count();
            if num_nodes != exp_num_nodes {
                return Err(InvalidSubcircuit::NotContiguous(interval.start_node()));
            }
        }

        let intervals = intervals
            .into_values()
            .map(|(interval, _)| interval)
            .collect_vec();

        let mut subcircuit = Self {
            intervals,
            input_copyable_values,
            input_resources: Vec::new(),
            output_resources: Vec::new(),
        };

        for res in subcircuit.resources().collect_vec() {
            subcircuit.update_input(res, circuit);
            subcircuit.update_output(res, circuit);
        }

        Ok(subcircuit)
    }

    /// Create a new empty subcircuit.
    pub fn new_empty() -> Self {
        Self::default()
    }

    /// Create a new subcircuit from a [`SiblingSubgraph`].
    pub fn try_from_subgraph(
        subgraph: &SiblingSubgraph<N>,
        circuit: &ResourceScope<impl HugrView<Node = N>>,
    ) -> Result<Self, InvalidSubcircuit<N>> {
        Self::try_from_nodes(subgraph.nodes().iter().copied(), circuit)
    }

    /// Extend the subcircuit to include the given node.
    ///
    /// Return whether the subcircuit whether the extension was successful, i.e.
    /// return `true` if the subcircuit was modified and `false` if it is left
    /// unchanged (because the node was already in the subcircuit).
    ///
    /// An error will be returned if the subcircuit cannot be extended to
    /// include the node. Currently, this also fails if the node has
    /// copyable values at its outputs.
    pub fn try_extend(
        &mut self,
        node: N,
        circuit: &ResourceScope<impl HugrView<Node = N>>,
    ) -> Result<bool, InvalidSubcircuit<N>> {
        // Do not support copyable values at node outputs
        let output_copyable_values = circuit.get_copyable_wires(node, Direction::Outgoing);
        if output_copyable_values.count() > 0 {
            return Err(InvalidSubcircuit::OutputCopyableValues(node));
        }

        let backup = self.to_owned();
        let mut was_changed = false;

        // Extend the subcircuit resource intervals to include the node
        match self.try_extend_resources(node, circuit) {
            Ok(new_change) => was_changed |= new_change,
            Err(err) => {
                *self = backup;
                return Err(err);
            }
        };

        // Add copyable inputs to the subcircuit where required
        was_changed |= self.extend_copyable_inputs(node, circuit);

        Ok(was_changed)
    }

    /// Iterate over the resources in the subcircuit.
    pub fn resources(&self) -> impl Iterator<Item = ResourceId> + '_ {
        self.intervals.iter().map(|interval| interval.resource_id())
    }

    /// Nodes in the subcircuit.
    pub fn nodes<'a>(
        &'a self,
        circuit: &'a ResourceScope<impl HugrView<Node = N>>,
    ) -> impl Iterator<Item = N> + 'a {
        self.intervals
            .iter()
            .map(|interval| circuit.nodes_in_interval(*interval))
            .kmerge_by(|&n1, &n2| {
                let pos1 = circuit.get_position(n1).expect("valid node");
                let pos2 = circuit.get_position(n2).expect("valid node");
                (pos1, n1) < (pos2, n2)
            })
            .dedup()
    }

    /// Number of nodes in the subcircuit.
    pub fn node_count(&self, circuit: &ResourceScope<impl HugrView<Node = N>>) -> usize {
        self.nodes(circuit).count()
    }

    /// Whether the subcircuit is empty.
    pub fn is_empty(&self) -> bool {
        self.intervals.is_empty()
    }

    /// Get the interval for the given line.
    pub fn get_interval(&self, resource: ResourceId) -> Option<Interval<N>> {
        self.intervals
            .iter()
            .find(|interval| interval.resource_id() == resource)
            .copied()
    }

    fn get_interval_mut(&mut self, resource: ResourceId) -> Option<&mut Interval<N>> {
        self.intervals
            .iter_mut()
            .find(|interval| interval.resource_id() == resource)
    }

    /// Iterate over the line indices of the subcircuit and their intervals.
    pub fn intervals_iter(&self) -> impl Iterator<Item = Interval<N>> + '_ {
        self.intervals.iter().copied()
    }

    /// Number of intervals in the subcircuit.
    pub fn num_intervals(&self) -> usize {
        self.intervals.len()
    }

    /// Convert the subcircuit to a [`SiblingSubgraph`].
    pub fn try_to_subgraph(
        &self,
        _circuit: &ResourceScope<impl HugrView<Node = N>>,
    ) -> Result<SiblingSubgraph<N>, InvalidSubgraph<N>> {
        todo!()
    }

    /// Create a rewrite rule to replace the subcircuit with a new circuit.
    ///
    /// # Parameters
    /// * `circuit` - The base circuit that contains the subcircuit.
    /// * `replacement` - The new circuit to replace the subcircuit with.
    pub fn create_rewrite(
        &self,
        replacement: Circuit<impl HugrView<Node = hugr::Node>>,
        circuit: &ResourceScope<impl HugrView<Node = N>>,
    ) -> Result<CircuitRewrite<N>, InvalidReplacement> {
        let hugr = circuit.hugr();
        let subgraph = self
            .try_to_subgraph(circuit)
            .map_err(|_| InvalidReplacement::NonConvexSubgraph)?;
        CircuitRewrite::try_new(&subgraph, hugr, replacement)
    }
}

fn update_copyable_inputs<N: HugrNode>(
    input_copyable_values: &mut Vec<Wire<N>>,
    node: N,
    circuit: &ResourceScope<impl HugrView<Node = N>>,
) -> Result<(), InvalidSubcircuit<N>> {
    for copyable_input in circuit.get_copyable_wires(node, Direction::Incoming) {
        if !input_copyable_values.contains(&copyable_input) {
            input_copyable_values.push(copyable_input);
        }
    }

    if circuit
        .get_copyable_wires(node, Direction::Outgoing)
        .count()
        > 0
    {
        return Err(InvalidSubcircuit::OutputCopyableValues(node));
    }

    Ok(())
}

/// Extend the intervals such that the given node is included.
fn extend_intervals<N: HugrNode>(
    intervals: &mut BTreeMap<ResourceId, (Interval<N>, usize)>,
    node: N,
    circuit: &ResourceScope<impl HugrView<Node = N>>,
) {
    for res in circuit.get_all_resources(node) {
        let (interval, num_nodes) = intervals
            .entry(res)
            .or_insert_with(|| (Interval::singleton(res, node, circuit), 0));
        let Some(pos) = circuit.get_position(node) else {
            panic!("node {node:?} is not on resource path {res:?}");
        };
        interval.add_node_unchecked(node, pos);
        *num_nodes += 1;
    }
}

// Private methods
impl<N: HugrNode> Subcircuit<N> {
    fn try_extend_resources(
        &mut self,
        node: N,
        circuit: &ResourceScope<impl HugrView<Node = N>>,
    ) -> Result<bool, InvalidSubcircuit<N>> {
        let mut was_changed = false;

        for resource_id in circuit.get_all_resources(node) {
            let interval = self.get_interval_mut(resource_id);
            if let Some(interval) = interval {
                match interval.try_extend(node, circuit) {
                    Ok(None) => { /* nothing to do */ }
                    Ok(Some(Direction::Incoming)) => {
                        // Added node to the left of the interval
                        was_changed = true;
                        self.update_input(resource_id, circuit);
                    }
                    Ok(Some(Direction::Outgoing)) => {
                        // Added node to the right of the interval
                        was_changed = true;
                        self.update_output(resource_id, circuit);
                    }
                    Err(InvalidInterval::NotContiguous(node)) => {
                        return Err(InvalidSubcircuit::NotContiguous(node));
                    }
                    Err(InvalidInterval::NotOnResourcePath(node)) => {
                        panic!("{resource_id:?} is not a valid resource for node {node:?}")
                    }
                }
            } else {
                was_changed = true;
                self.intervals
                    .push(Interval::singleton(resource_id, node, circuit));
                self.update_input(resource_id, circuit);
                self.update_output(resource_id, circuit);
            }
        }

        Ok(was_changed)
    }

    fn extend_copyable_inputs(
        &mut self,
        node: N,
        circuit: &ResourceScope<impl HugrView<Node = N>>,
    ) -> bool {
        let input_copyable_values = circuit.get_copyable_wires(node, Direction::Incoming);
        let mut was_changed = false;

        for copyable_value_id in input_copyable_values {
            if !self.input_copyable_values.contains(&copyable_value_id) {
                self.input_copyable_values.push(copyable_value_id);
                was_changed = true;
            }
        }

        was_changed
    }

    /// Whether the resource should be part of the incoming/outgoing subcircuit
    /// boundary.
    fn is_in_boundary(
        &self,
        resource_id: ResourceId,
        circuit: &ResourceScope<impl HugrView<Node = N>>,
        dir: Direction,
    ) -> bool {
        let Some(interval) = self.get_interval(resource_id) else {
            return false;
        };
        let node = match dir {
            Direction::Incoming => interval.start_node(),
            Direction::Outgoing => interval.end_node(),
        };
        circuit.get_port(node, resource_id, dir).is_some()
    }

    fn update_input(
        &mut self,
        resource_id: ResourceId,
        circuit: &ResourceScope<impl HugrView<Node = N>>,
    ) {
        if self.is_in_boundary(resource_id, circuit, Direction::Incoming) {
            if !self.input_resources.contains(&resource_id) {
                self.input_resources.push(resource_id);
            }
        } else {
            self.input_resources.retain(|id| *id != resource_id);
        }
    }

    fn update_output(
        &mut self,
        resource_id: ResourceId,
        circuit: &ResourceScope<impl HugrView<Node = N>>,
    ) {
        if self.is_in_boundary(resource_id, circuit, Direction::Outgoing) {
            if !self.output_resources.contains(&resource_id) {
                self.output_resources.push(resource_id);
            }
        } else {
            self.output_resources.retain(|id| *id != resource_id);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        resource::{
            tests::{cx_circuit, cx_rz_circuit},
            ResourceAllocator,
        },
        utils::build_simple_circuit,
        TketOp,
    };
    use hugr::{CircuitUnit, Hugr, Node};
    use rstest::{fixture, rstest};

    #[rstest]
    #[case::empty_set(vec![], true, "empty subcircuit is valid")]
    #[case::single_node(vec![0], true, "single node should succeed")]
    #[case::two_adjacent_nodes(vec![0, 1], true, "two adjacent nodes should succeed")]
    #[case::three_adjacent_nodes(vec![0, 1, 2], true, "three adjacent nodes should succeed")]
    #[case::all_nodes(vec![0, 1, 2, 3, 4], true, "all nodes should succeed")]
    #[case::non_adjacent_nodes(vec![0, 2], false, "non-adjacent nodes should fail")]
    #[case::gap_in_middle(vec![0, 1, 3, 4], false, "gap in middle should fail")]
    #[case::last_two_nodes(vec![3, 4], true, "last two nodes should succeed")]
    fn test_try_from_nodes_cx_circuit(
        #[case] node_indices: Vec<usize>,
        #[case] should_succeed: bool,
        #[case] description: &str,
    ) {
        let circ = cx_circuit(5);
        let subgraph = Circuit::from(&circ).subgraph();
        let cx_nodes = subgraph.nodes().to_owned();
        let scope = ResourceScope::new(&circ, subgraph);

        let nodes: Vec<_> = node_indices.into_iter().map(|i| cx_nodes[i]).collect();

        let result = Subcircuit::try_from_nodes(nodes.iter().copied(), &scope);

        if should_succeed {
            assert!(result.is_ok(), "Expected success for case: {description}");
            let subcircuit = result.unwrap();
            assert_eq!(subcircuit.nodes(&scope).collect_vec(), nodes);
        } else {
            assert!(result.is_err(), "Expected failure for case: {description}");
        }
    }

    #[rstest]
    #[case::empty_set(vec![], true, 0, 0, 0)]
    #[case::singe_h_gate(vec![7], true, 1, 1, 0)]
    #[case::two_h_gates(vec![7, 8], true, 2, 2, 0)]
    #[case::h_and_cx_gate(vec![7, 9], true, 2, 2, 0)]
    #[case::cx_rz_rz_same_angle(vec![9, 10, 11], true, 2, 2, 1)]
    #[case::cx_rz_rz_diff_angle(vec![9, 10, 15], true, 2, 2, 2)]
    fn test_try_from_nodes_cx_rz_circuit(
        #[case] node_indices: Vec<usize>,
        #[case] should_succeed: bool,
        #[case] expected_input_resources: usize,
        #[case] expected_output_resources: usize,
        #[case] expected_copyable_inputs: usize,
    ) {
        let circ = cx_rz_circuit(2, true, true);
        let subgraph = Circuit::from(&circ).subgraph();
        let scope = ResourceScope::new(&circ, subgraph);

        let selected_nodes: Vec<_> = node_indices
            .into_iter()
            .map(|i| Node::from(portgraph::NodeIndex::new(i)))
            .collect();

        let result = Subcircuit::try_from_nodes(selected_nodes.iter().copied(), &scope);

        if should_succeed {
            assert!(result.is_ok());
            let subcircuit = result.unwrap();
            assert_eq!(subcircuit.nodes(&scope).collect_vec(), selected_nodes);
            assert_eq!(
                subcircuit.input_resources.len(),
                expected_input_resources,
                "Wrong number of input resources"
            );
            assert_eq!(
                subcircuit.output_resources.len(),
                expected_output_resources,
                "Wrong number of output resources"
            );
            assert_eq!(
                subcircuit.input_copyable_values.len(),
                expected_copyable_inputs,
                "Wrong number of copyable inputs"
            );
        } else {
            assert!(result.is_err());
        }
    }

    #[test]
    fn try_extend_cx_rz_circuit() {
        let circ = cx_rz_circuit(2, true, true);
        let subgraph = Circuit::from(&circ).subgraph();
        let circ = ResourceScope::new(circ, subgraph);

        let mut subcircuit = Subcircuit::new_empty();

        let node = |i: usize| Node::from(portgraph::NodeIndex::new(i));
        let resources = {
            let mut alloc = ResourceAllocator::new();
            [alloc.allocate(), alloc.allocate()]
        };

        // Add first a H gate
        assert_eq!(subcircuit.try_extend(node(7), &circ), Ok(true));
        assert_eq!(subcircuit.resources().collect_vec(), [resources[0]]);
        assert_eq!(subcircuit.input_resources, [resources[0]]);
        assert_eq!(subcircuit.output_resources, [resources[0]]);
        assert_eq!(subcircuit.try_extend(node(7), &circ), Ok(false));

        // Now add a two-qubit CX gate
        assert_eq!(subcircuit.try_extend(node(9), &circ), Ok(true));
        assert_eq!(subcircuit.resources().collect_vec(), resources);
        assert_eq!(subcircuit.input_resources, resources);
        assert_eq!(subcircuit.output_resources, resources);
        assert_eq!(subcircuit.input_copyable_values, vec![]);
        assert_eq!(subcircuit.try_extend(node(9), &circ), Ok(false));

        // Cannot add this non-contiguous rotation
        let subcircuit_clone = subcircuit.clone();
        assert_eq!(
            subcircuit.try_extend(node(16), &circ),
            Err(InvalidSubcircuit::NotContiguous(node(16)))
        );
        assert_eq!(subcircuit, subcircuit_clone);

        // Now add a contiguous rotation
        assert_eq!(subcircuit.try_extend(node(10), &circ), Ok(true));
        assert_eq!(subcircuit.resources().collect_vec(), resources);
        assert_eq!(subcircuit.input_resources, resources);
        assert_eq!(subcircuit.output_resources, resources);
        assert_eq!(subcircuit.input_copyable_values.len(), 1);
        assert_eq!(subcircuit.try_extend(node(10), &circ), Ok(false));

        // One more rotation, same angle
        assert_eq!(subcircuit.try_extend(node(11), &circ), Ok(true));
        assert_eq!(subcircuit.resources().collect_vec(), resources);
        assert_eq!(subcircuit.input_resources, resources);
        assert_eq!(subcircuit.output_resources, resources);
        assert_eq!(subcircuit.input_copyable_values.len(), 1);
        assert_eq!(subcircuit.try_extend(node(11), &circ), Ok(false));

        // Last rotation, different angle
        // now the previously non-contiguous rotation is contiguous
        assert_eq!(subcircuit.try_extend(node(16), &circ), Ok(true));
        assert_eq!(subcircuit.resources().collect_vec(), resources);
        assert_eq!(subcircuit.input_resources, resources);
        assert_eq!(subcircuit.output_resources, resources);
        assert_eq!(subcircuit.input_copyable_values.len(), 2);
        assert_eq!(subcircuit.try_extend(node(16), &circ), Ok(false));

        assert_eq!(subcircuit.node_count(&circ), 5);
    }

    #[fixture]
    fn ancilla_circ() -> ResourceScope<Hugr> {
        let circ = build_simple_circuit(1, |circ| {
            let empty: [CircuitUnit; 0] = []; // requires type annotation
            let ancilla = circ.append_with_outputs(TketOp::QAlloc, empty)?[0];

            let ancilla = circ.append_with_outputs(
                TketOp::CX,
                [CircuitUnit::Linear(0), CircuitUnit::Wire(ancilla)],
            )?[0];
            circ.append_and_consume(TketOp::QFree, [ancilla])?;

            Ok(())
        })
        .unwrap();
        let subgraph = circ.subgraph();
        ResourceScope::new(circ.into_hugr(), subgraph)
    }

    #[rstest]
    fn try_extend_remove_input_output(ancilla_circ: ResourceScope<Hugr>) {
        let mut subcircuit = Subcircuit::new_empty();
        let node = |i: usize| Node::from(portgraph::NodeIndex::new(i));
        let resources = {
            let mut alloc = ResourceAllocator::new();
            [alloc.allocate(), alloc.allocate()]
        };

        // Add a two-qubit CX gates, as usual => two inputs, two outputs
        assert_eq!(subcircuit.try_extend(node(5), &ancilla_circ), Ok(true));
        assert_eq!(subcircuit.resources().collect_vec(), resources);
        assert_eq!(subcircuit.input_resources, resources);
        assert_eq!(subcircuit.output_resources, resources);
        assert_eq!(subcircuit.input_copyable_values, vec![]);

        // Add the qalloc; now the second qubit is no more an input
        assert_eq!(subcircuit.try_extend(node(4), &ancilla_circ), Ok(true));
        assert_eq!(subcircuit.resources().collect_vec(), resources);
        assert_eq!(subcircuit.input_resources, [resources[0]]);
        assert_eq!(subcircuit.output_resources, resources);
        assert_eq!(subcircuit.input_copyable_values, vec![]);

        // Add the qfree; the second qubit is no longer an output either
        assert_eq!(subcircuit.try_extend(node(6), &ancilla_circ), Ok(true));
        assert_eq!(subcircuit.resources().collect_vec(), resources);
        assert_eq!(subcircuit.input_resources, [resources[0]]);
        assert_eq!(subcircuit.output_resources, [resources[0]]);
        assert_eq!(subcircuit.input_copyable_values, vec![]);
    }
}
