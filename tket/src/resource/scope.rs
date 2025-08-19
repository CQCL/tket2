//! Main ResourceScope implementation for tracking resources in HUGR subgraphs.
//!
//! This module provides the ResourceScope struct which manages resource
//! tracking within a specific region of a HUGR, computing resource paths and
//! providing efficient lookup of port values.

use crate::resource::flow::{DefaultResourceFlow, ResourceFlow};
use crate::resource::types::{CopyableValueId, OpValue, PortMap};
use crate::utils::type_is_linear;
use hugr::hugr::views::SiblingSubgraph;
use hugr::ops::OpTrait;
use hugr::{Direction, HugrView, IncomingPort, Port, PortIndex};
use hugr_core::hugr::internal::PortgraphNodeMap;
use indexmap::IndexMap;
use itertools::Itertools;
use portgraph::algorithms::{toposort, TopoSort};
use portgraph::view::{FilteredGraph, NodeFilter, NodeFiltered};

use super::types::{PositionAllocator, ResourceMap};
use super::ResourceAllocator;

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
    op_values: IndexMap<H::Node, PortMap<OpValue>>,
}

/// Configuration for a ResourceScope.
pub struct ResourceScopeConfig {
    flows: Vec<Box<dyn ResourceFlow>>,
}

impl Default for ResourceScopeConfig {
    fn default() -> Self {
        Self {
            flows: vec![Box::new(DefaultResourceFlow::new())],
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
        config: &ResourceScopeConfig,
    ) -> Self {
        let mut scope = Self {
            hugr,
            subgraph,
            op_values: IndexMap::new(),
        };
        scope.compute_op_values(&config.flows);
        scope
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
    pub fn get_opvalue(&self, node: H::Node, port: impl Into<Port>) -> &OpValue {
        let port_map = self.op_values.get(&node).expect("node not in scope");
        port_map.get(port)
    }

    /// Get all opvalues for either the incoming or outgoing ports of a node.
    pub fn get_opvalue_slice(&self, node: H::Node, direction: Direction) -> &[OpValue] {
        let port_map = self.op_values.get(&node).expect("node not in scope");
        port_map.get_slice(direction)
    }
}

// Private methods to construct the op_values map.
impl<H: HugrView> ResourceScope<H> {
    /// Compute op values for all nodes in the subgraph.
    fn compute_op_values(&mut self, flows: &[Box<dyn ResourceFlow>]) {
        let mut allocator = Allocator::default();

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
        allocator: &mut Allocator,
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
                let port_map = self.op_values.entry(node).or_insert_with(|| {
                    let signature = self
                        .hugr
                        .get_optype(node)
                        .dataflow_signature()
                        .expect("dataflow op");
                    PortMap::with_default(sentinel, &signature)
                });
                port_map.set(port, op_value);
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
        allocator: &mut Allocator,
        sentinel: OpValue,
    ) {
        let signature = self
            .hugr
            .get_optype(node)
            .dataflow_signature()
            .expect("dataflow op");
        let port_map = self
            .op_values
            .entry(node)
            .or_insert_with(|| PortMap::with_default(sentinel, &signature));
        for p in signature.input_ports() {
            if port_map.get(p) == &sentinel {
                let op_value = allocator.allocate_op_value(node, p, &self.hugr);
                port_map.set(p, op_value);
            }
        }
    }

    /// Propagate op values at input ports to output ports.
    fn propagate_to_outputs(
        &mut self,
        node: H::Node,
        flows: &[Box<dyn ResourceFlow>],
        allocator: &mut Allocator,
    ) {
        let port_map = self.op_values.get_mut(&node).expect("known node");

        let inp_resources = port_map
            .get_slice(Direction::Incoming)
            .iter()
            .map(|&op_val| match op_val {
                OpValue::Resource(res, _) => Some(res),
                OpValue::Copyable(_) => None,
            })
            .collect_vec();

        let out_resources = flows
            .iter()
            .find_map(|f| {
                f.map_resources(self.hugr.get_optype(node), &inp_resources)
                    .ok()
            })
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

        for p in signature.output_ports() {
            let op_value = *self.get_opvalue(node, p);

            for (in_node, in_port) in self.hugr.linked_inputs(node, p) {
                if !self.subgraph.nodes().contains(&in_node) {
                    continue;
                }
                let op = self.hugr.get_optype(in_node);
                let Some(signature) = op.dataflow_signature() else {
                    continue;
                };
                let next_port_map = self
                    .op_values
                    .entry(in_node)
                    .or_insert_with(|| PortMap::with_default(sentinel, &signature));
                let next_op_value = match op_value {
                    OpValue::Resource(id, pos) => OpValue::Resource(id, pos.increment()),
                    copyable @ OpValue::Copyable(_) => copyable,
                };
                next_port_map.set(in_port, next_op_value);
            }
        }
    }
}

#[derive(Debug, Clone, Default)]
struct Allocator {
    /// Allocator for new resource IDs.
    resource: ResourceAllocator,
    /// For each resource ID, the allocator for new positions on the resource
    /// path.
    position: ResourceMap<PositionAllocator>,
}

impl Allocator {
    fn allocate_resource(&mut self) -> OpValue {
        let resource_id = self.resource.allocate();
        if resource_id.as_usize() >= self.position.len() {
            self.position
                .resize(resource_id.as_usize() + 1, PositionAllocator::default());
        }
        let position = self.position[resource_id.as_usize()].allocate();
        OpValue::Resource(resource_id, position)
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
    use crate::resource::{Position, ResourceId};

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

            for (&node, port_map) in &scope.op_values {
                for (port, &op_value) in port_map.iter() {
                    match op_value {
                        OpValue::Resource(res, pos) => resource_paths
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
}
