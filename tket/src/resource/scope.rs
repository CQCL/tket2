//! Main ResourceScope implementation for tracking resources in HUGR subgraphs.
//!
//! This module provides the ResourceScope struct which manages resource
//! tracking within a specific region of a HUGR, computing resource paths and
//! providing efficient lookup of port values.

use std::{cmp, iter, mem};

use crate::resource::flow::{DefaultResourceFlow, ResourceFlow};
use crate::resource::types::{CopyableValueId, OpValue, PortMap};
use crate::utils::type_is_linear;
use crate::Circuit;
use hugr::hugr::views::SiblingSubgraph;
use hugr::ops::OpTrait;
use hugr::types::Signature;
use hugr::{Direction, HugrView, IncomingPort, Port, PortIndex};
use hugr_core::hugr::internal::PortgraphNodeMap;
use indexmap::IndexMap;
use itertools::Itertools;
use portgraph::algorithms::{toposort, TopoSort};
use portgraph::view::{FilteredGraph, NodeFilter, NodeFiltered};

use super::{Position, ResourceAllocator, ResourceId};

/// ResourceScope tracks resources within a HUGR subgraph.
///
/// This struct computes and caches resource paths through a given subgraph,
/// allowing efficient lookup of OpValues for any port of any operation within
/// the scope.
#[derive(Debug, Clone)]
pub struct ResourceScope<H: HugrView> {
    /// The HUGR containing the operations.
    hugr: H,
    /// The subgraph within which resources are tracked.
    subgraph: SiblingSubgraph<H::Node>,
    /// Mapping from nodes and ports to their OpValues.
    op_values: IndexMap<H::Node, NodeOpValues>,
}

#[derive(Debug, Clone)]
struct NodeOpValues {
    /// Mapping from ports to their OpValues.
    port_map: PortMap<OpValue>,
    /// The position of the node.
    position: Position,
}

impl NodeOpValues {
    fn with_default(default: OpValue, signature: &Signature) -> Self {
        Self {
            port_map: PortMap::with_default(default, signature),
            position: Position::default(),
        }
    }
}

/// Configuration for a ResourceScope.
pub struct ResourceScopeConfig<'a, H: HugrView> {
    flows: Vec<Box<dyn 'a + ResourceFlow<H>>>,
}

impl<H: HugrView> Default for ResourceScopeConfig<'_, H> {
    fn default() -> Self {
        Self {
            flows: vec![Box::new(DefaultResourceFlow::new())],
        }
    }
}

impl<'a, H: HugrView, RF: ResourceFlow<H> + 'a> FromIterator<RF> for ResourceScopeConfig<'a, H> {
    fn from_iter<T: IntoIterator<Item = RF>>(iter: T) -> Self {
        Self {
            flows: iter
                .into_iter()
                .map(|rf| Box::new(rf) as Box<dyn ResourceFlow<H>>)
                .collect(),
        }
    }
}

impl<H: HugrView> ResourceScope<H> {
    /// Create a new ResourceScope from a SiblingSubgraph using the default
    /// resource flow.
    pub fn new(hugr: H, subgraph: SiblingSubgraph<H::Node>) -> Self {
        Self::with_config(hugr, subgraph, &Default::default())
    }

    /// Create a new ResourceScope with a custom resource flow implementation.
    pub fn with_config(
        hugr: H,
        subgraph: SiblingSubgraph<H::Node>,
        config: &ResourceScopeConfig<H>,
    ) -> Self {
        let mut scope = Self {
            hugr,
            subgraph,
            op_values: IndexMap::new(),
        };
        scope.compute_op_values(&config.flows);
        scope
    }

    /// Get the nodes within the scope.
    pub fn nodes(&self) -> &[H::Node] {
        self.subgraph.nodes()
    }

    /// Get the underlying HUGR.
    pub fn hugr(&self) -> &H {
        &self.hugr
    }

    /// Get the underlying subgraph.
    pub fn subgraph(&self) -> &SiblingSubgraph<H::Node> {
        &self.subgraph
    }

    /// Get the opvalue for a given port.
    pub fn get_opvalue(&self, node: H::Node, port: impl Into<Port>) -> Option<OpValue> {
        let port_map = self.port_map(node)?;
        Some(*port_map.get(port))
    }

    /// Get all opvalues for either the incoming or outgoing ports of a node.
    pub fn get_opvalue_slice(&self, node: H::Node, direction: Direction) -> Option<&[OpValue]> {
        let port_map = self.port_map(node)?;
        Some(port_map.get_slice(direction))
    }

    /// Get the port of node on the given resource path.
    ///
    /// The returned port will have the direction `dir`.
    pub fn get_port(&self, node: H::Node, resource_id: ResourceId, dir: Direction) -> Option<Port> {
        let opvals = self.get_opvalue_slice(node, dir)?;
        let offset = opvals.iter().position(|opval| match opval {
            &OpValue::Resource(res) => res == resource_id,
            _ => false,
        })?;
        Some(Port::new(dir, offset))
    }

    /// Get the position of the given node.
    pub fn get_position(&self, node: H::Node) -> Option<Position> {
        self.op_values
            .get(&node)
            .map(|node_op_values| node_op_values.position)
    }

    /// All resource IDs on the ports of `node` in the given direction.
    pub fn get_resources(
        &self,
        node: H::Node,
        dir: Direction,
    ) -> impl Iterator<Item = ResourceId> + '_ {
        let opvals = self.get_opvalue_slice(node, dir);
        opvals
            .into_iter()
            .flatten()
            .filter_map(|opval| match opval {
                &OpValue::Resource(res) => Some(res),
                _ => None,
            })
    }

    /// All resource IDs on the ports of `node`, in both directions.
    pub fn get_all_resources(&self, node: H::Node) -> Vec<ResourceId> {
        let in_resources = self.get_resources(node, Direction::Incoming);
        let out_resources = self.get_resources(node, Direction::Outgoing);
        let mut all_resources = in_resources.chain(out_resources).collect_vec();
        all_resources.sort_unstable();
        all_resources.dedup();
        all_resources.shrink_to_fit();
        all_resources
    }

    /// Whether the given node is the first node on the path of the given resource.
    pub fn is_resource_start(&self, node: H::Node, resource_id: ResourceId) -> bool {
        self.get_port(node, resource_id, Direction::Outgoing)
            .is_some()
            && self
                .get_port(node, resource_id, Direction::Incoming)
                .is_none()
    }

    /// Iterate over all resources in the scope.
    pub fn resources_iter(&self) -> impl Iterator<Item = ResourceId> + '_ {
        self.nodes()
            .iter()
            .map(|&n| self.get_all_resources(n))
            .kmerge()
            .dedup()
    }

    /// All copyable values on the ports of `node` in the given direction.
    pub fn get_copyable_values(
        &self,
        node: H::Node,
        dir: Direction,
    ) -> impl Iterator<Item = CopyableValueId> + '_ {
        let opvals = self.get_opvalue_slice(node, dir);
        opvals
            .into_iter()
            .flatten()
            .filter_map(|opval| match opval {
                &OpValue::Copyable(id) => Some(id),
                _ => None,
            })
    }

    /// Iterate over the nodes on the resource path starting from the given
    /// node in the given direction.
    pub fn resource_path_iter(
        &self,
        resource_id: ResourceId,
        start_node: H::Node,
        direction: Direction,
    ) -> impl Iterator<Item = H::Node> + '_ {
        let mut curr_node = start_node;

        iter::from_fn(move || {
            let port = self.get_port(curr_node, resource_id, direction)?;
            let (next_node, _) = self
                .hugr()
                .single_linked_port(curr_node, port)
                .expect("linear resource");

            Some(mem::replace(&mut curr_node, next_node))
        })
    }
}

impl<H: HugrView> ResourceScope<H> {
    /// Create a new ResourceScope from a reference to a circuit.
    pub fn from_circuit(circuit: Circuit<H>) -> Self
    where
        H: Clone + HugrView<Node = hugr::Node>,
    {
        let subgraph = circuit.subgraph();
        Self::new(circuit.into_hugr(), subgraph)
    }
}

impl<'h, H: HugrView> ResourceScope<&'h H> {
    /// Create a new ResourceScope from a reference to a circuit.
    pub fn from_circuit_ref(circuit: &'h Circuit<H>) -> Self
    where
        H: Clone + HugrView<Node = hugr::Node>,
    {
        let subgraph = circuit.subgraph();
        Self::new(circuit.hugr(), subgraph)
    }
}

impl<H: HugrView<Node = hugr::Node> + Clone> From<Circuit<H>> for ResourceScope<H> {
    fn from(value: Circuit<H>) -> Self {
        Self::from_circuit(value)
    }
}

impl<'h, H: HugrView<Node = hugr::Node> + Clone> From<&'h Circuit<H>> for ResourceScope<&'h H> {
    fn from(value: &'h Circuit<H>) -> Self {
        Self::from_circuit_ref(value)
    }
}

// Private methods to construct the op_values map.
impl<H: HugrView> ResourceScope<H> {
    fn port_map(&self, node: H::Node) -> Option<&PortMap<OpValue>> {
        Some(&self.op_values.get(&node)?.port_map)
    }

    /// Compute op values for all nodes in the subgraph.
    fn compute_op_values(&mut self, flows: &[Box<dyn '_ + ResourceFlow<H>>]) {
        let mut allocator = OpValueAllocator::default();

        // Sentinel value for uninitialized ports
        let sentinel = OpValue::Copyable(CopyableValueId::new());

        // First, assign op values to the inputs to the subgraph.
        self.assign_op_values(
            self.subgraph.incoming_ports().to_owned(),
            &mut allocator,
            sentinel,
        );

        // Proceed to propagating the op values through the subgraph, in topological
        // order.
        for node in toposort_subgraph(&self.hugr, &self.subgraph, self.find_sources()) {
            if self.hugr.get_optype(node).dataflow_signature().is_none() {
                // ignore non-dataflow ops
                continue;
            }
            self.assign_missing_op_values(node, &mut allocator, sentinel);
            self.propagate_to_outputs(node, flows, &mut allocator);
            self.propagate_to_next_inputs(node, sentinel);
        }
    }

    /// Assign op values to the given port groups.
    ///
    /// The ports are partitioned into sets of ports connected to a same value.
    /// The port sets can be non-singleton only if the value is copyable.
    fn assign_op_values<I: IntoIterator<Item = (H::Node, IncomingPort)>>(
        &mut self,
        port_groups: impl IntoIterator<Item = I>,
        allocator: &mut OpValueAllocator,
        sentinel: OpValue,
    ) {
        for all_uses_of_value in port_groups {
            let mut all_uses_of_input = all_uses_of_value.into_iter().peekable();
            let Some(&(fst_node, fst_port)) = all_uses_of_input.peek() else {
                // this input is not used in the subgraph, we can skip it
                continue;
            };
            // We allocate one opvalue for all uses of this input
            let op_value = allocator.allocate_op_value(fst_node, fst_port, &self.hugr);
            for (node, port) in all_uses_of_input {
                let node_op_values = self.op_values.entry(node).or_insert_with(|| {
                    let signature = self
                        .hugr
                        .get_optype(node)
                        .dataflow_signature()
                        .expect("dataflow op");
                    NodeOpValues::with_default(sentinel, &signature)
                });
                node_op_values.port_map.set(port, op_value);
            }
        }
    }

    /// Find source nodes (nodes with no predecessors in the subgraph).
    fn find_sources(&self) -> impl Iterator<Item = H::Node> + '_ {
        let has_pred_in_subgraph = |node: H::Node| {
            self.hugr
                .all_linked_outputs(node)
                .any(|(n, _)| self.subgraph.nodes().contains(&n))
        };

        self.subgraph
            .nodes()
            .iter()
            .copied()
            .filter(move |&n| !has_pred_in_subgraph(n))
    }

    /// Ensure all input ports have assigned op values.
    fn assign_missing_op_values(
        &mut self,
        node: H::Node,
        allocator: &mut OpValueAllocator,
        sentinel: OpValue,
    ) {
        let signature = self
            .hugr
            .get_optype(node)
            .dataflow_signature()
            .expect("dataflow op");
        let node_op_values = self
            .op_values
            .entry(node)
            .or_insert_with(|| NodeOpValues::with_default(sentinel, &signature));
        for p in signature.input_ports() {
            if node_op_values.port_map.get(p) == &sentinel {
                let op_value = allocator.allocate_op_value(node, p, &self.hugr);
                node_op_values.port_map.set(p, op_value);
            }
        }
    }

    /// Propagate op values at input ports to output ports.
    fn propagate_to_outputs(
        &mut self,
        node: H::Node,
        flows: &[Box<dyn '_ + ResourceFlow<H>>],
        allocator: &mut OpValueAllocator,
    ) {
        let port_map = &mut self.op_values.get_mut(&node).expect("known node").port_map;

        let inp_resources = port_map
            .get_slice(Direction::Incoming)
            .iter()
            .map(|&op_val| match op_val {
                OpValue::Resource(res) => Some(res),
                OpValue::Copyable(_) => None,
            })
            .collect_vec();

        let out_resources = flows
            .iter()
            .find_map(|f| f.map_resources(node, &self.hugr, &inp_resources).ok())
            .expect("no flow found");

        let signature = self
            .hugr
            .get_optype(node)
            .dataflow_signature()
            .expect("dataflow op");
        // Set out resources to output, create new opvalues where required
        for p in signature.output_ports() {
            let op_value = match out_resources.get(p.index()).copied().flatten() {
                Some(resource_id) => {
                    let index = inp_resources
                        .iter()
                        .position(|&res| res == Some(resource_id))
                        .expect("invalid resource ID returned by flow");
                    *port_map.get(IncomingPort::from(index))
                }
                None => allocator.allocate_op_value(node, p, &self.hugr),
            };
            port_map.set(p, op_value);
        }
    }

    /// Propagate op values at output ports across wires to connected inputs.
    fn propagate_to_next_inputs(&mut self, node: H::Node, sentinel: OpValue) {
        let signature = self
            .hugr
            .get_optype(node)
            .dataflow_signature()
            .expect("dataflow op");
        let pos = self.get_position(node).expect("known node");

        for p in signature.output_ports() {
            let op_value = self.get_opvalue(node, p).expect("known node");

            for (in_node, in_port) in self.hugr.linked_inputs(node, p) {
                if !self.subgraph.nodes().contains(&in_node) {
                    continue;
                }
                let op = self.hugr.get_optype(in_node);
                let Some(signature) = op.dataflow_signature() else {
                    continue;
                };
                let next_node_op_values = self
                    .op_values
                    .entry(in_node)
                    .or_insert_with(|| NodeOpValues::with_default(sentinel, &signature));
                let next_op_value = op_value;
                next_node_op_values.port_map.set(in_port, next_op_value);
                next_node_op_values.position =
                    cmp::max(next_node_op_values.position, pos.increment());
            }
        }
    }
}

#[derive(Debug, Clone, Default)]
struct OpValueAllocator(ResourceAllocator);

impl OpValueAllocator {
    fn allocate_resource(&mut self) -> OpValue {
        let resource_id = self.0.allocate();
        OpValue::Resource(resource_id)
    }

    fn allocate_copyable(&mut self) -> OpValue {
        OpValue::Copyable(CopyableValueId::new())
    }

    fn allocate_op_value<H: HugrView>(
        &mut self,
        node: H::Node,
        port: impl Into<Port>,
        hugr: &H,
    ) -> OpValue {
        let op = hugr.get_optype(node);
        let signature = op.dataflow_signature().expect("dataflow op");
        let port = port.into();
        let ty = match port.direction() {
            Direction::Incoming => &signature.input()[port.index()],
            Direction::Outgoing => &signature.output()[port.index()],
        };
        if type_is_linear(ty) {
            self.allocate_resource()
        } else {
            self.allocate_copyable()
        }
    }
}

fn toposort_subgraph<'h, H: HugrView>(
    hugr: &'h H,
    subgraph: &'h SiblingSubgraph<H::Node>,
    sources: impl IntoIterator<Item = H::Node>,
) -> Vec<H::Node> {
    fn contains_node<H: HugrView>(
        node: portgraph::NodeIndex,
        (subgraph, pg_map): &(&SiblingSubgraph<H::Node>, &H::RegionPortgraphNodes),
    ) -> bool {
        subgraph.nodes().contains(&pg_map.from_portgraph(node))
    }

    let (pg, pg_map) = hugr.region_portgraph(subgraph.get_parent(hugr));
    let pg: NodeFiltered<_, NodeFilter<_>, _> =
        FilteredGraph::new(&pg, contains_node::<H>, |_, _| true, (subgraph, &pg_map));
    let topo: TopoSort<_> = toposort(
        pg,
        sources.into_iter().map(|n| pg_map.to_portgraph(n)),
        Direction::Outgoing,
    );

    topo.map(|n| pg_map.from_portgraph(n)).collect()
}

#[cfg(test)]
pub(crate) mod tests {
    //! Implementation of [`ResourceScopeReport`], used for testing.
    //!
    //! Tests are found in the parent module.

    use super::*;
    use std::collections::{BTreeMap, HashSet};

    use hugr::HugrView;

    use super::{OpValue, ResourceScope};
    use crate::{
        resource::{Position, ResourceId},
        utils::build_simple_circuit,
        TketOp,
    };

    pub type PathEl<N> = (Position, N, Port);

    /// Statistics about a ResourceScope.
    #[derive(Debug, Clone)]
    pub struct ResourceScopeReport<H: HugrView> {
        pub hugr: H,
        pub resource_paths: BTreeMap<ResourceId, Vec<PathEl<H::Node>>>,
        pub n_copyable: usize,
    }

    impl<'a, H: HugrView> From<&'a ResourceScope<H>> for ResourceScopeReport<&'a H> {
        fn from(scope: &'a ResourceScope<H>) -> Self {
            let mut resource_paths: BTreeMap<ResourceId, Vec<PathEl<H::Node>>> = BTreeMap::new();
            let mut copyable_values = HashSet::new();

            for (&node, op_values) in &scope.op_values {
                let pos = op_values.position;
                for (port, &op_value) in op_values.port_map.iter() {
                    match op_value {
                        OpValue::Resource(res) => resource_paths
                            .entry(res)
                            .or_default()
                            .push((pos, node, port)),
                        OpValue::Copyable(id) => {
                            copyable_values.insert(id);
                        }
                    }
                }
            }

            for path in resource_paths.values_mut() {
                path.sort_unstable();
            }

            Self {
                hugr: &scope.hugr,
                resource_paths,
                n_copyable: copyable_values.len(),
            }
        }
    }

    impl<H: HugrView> std::fmt::Display for ResourceScopeReport<H> {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            writeln!(f, "Found {} copyable values", self.n_copyable)?;
            writeln!(f, "Found {} resource paths:", self.resource_paths.len())?;
            for (res, path) in &self.resource_paths {
                writeln!(f, "  - {res:?}:")?;
                let mut path = path.iter().peekable();
                while let Some(&(pos, node, port)) = path.next() {
                    if let Some(&&(next_pos, next_node, next_port)) = path.peek() {
                        if next_node == node {
                            debug_assert_eq!(pos, next_pos);
                            path.next();
                            let in_port = port.as_incoming().unwrap();
                            let out_port = next_port.as_outgoing().unwrap();
                            let op_desc = self.hugr.get_optype(node).description();
                            writeln!(f, "    * {op_desc}({node:?}) [{in_port} -> {out_port}]",)?;
                            continue;
                        }
                    }
                    writeln!(f, "    * {node:?}@{} [{port}]", pos.to_f64(2))?;
                }
                writeln!(f)?;
            }
            Ok(())
        }
    }

    #[test]
    fn test_position_monotonic() {
        const N_HADAMARDS: [usize; 3] = [4, 10, 1];
        // A circuit with 3 qubits and a certain number of H on each qubit
        let circ = build_simple_circuit(3, |circ| {
            for (qb, n_hadamards) in N_HADAMARDS.iter().enumerate() {
                for _ in 0..*n_hadamards {
                    circ.append(TketOp::H, [qb])?;
                }
            }
            Ok(())
        })
        .unwrap();

        let scope = ResourceScope::from(&circ);
        let first_hadamards = circ
            .hugr()
            .all_linked_inputs(circ.input_node())
            .map(|(n, _)| n);

        for h in first_hadamards {
            let res = scope
                .get_all_resources(h)
                .into_iter()
                .exactly_one()
                .unwrap();
            let nodes_on_path = scope.resource_path_iter(res, h, Direction::Outgoing);
            let pos_on_path = nodes_on_path.map(|n| scope.get_position(n).unwrap());

            assert!(
                pos_on_path.collect_vec().windows(2).all(|w| w[0] <= w[1]),
                "position is not monotonically increasing on path {res:?}"
            );
        }
    }
}
