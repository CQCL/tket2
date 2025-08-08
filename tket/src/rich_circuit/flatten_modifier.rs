use std::collections::HashMap;

/// Take the fixed point of modifer calls.
///
use crate::circuit::Circuit;
use crate::extension::{TKET2_EXTENSION, TKET2_EXTENSION_ID as EXTENSION_ID};
use crate::rich_circuit::Modifier;
use hugr::algorithms::call_graph::{CallGraphEdge, CallGraphNode};
use hugr::core::HugrNode;
use hugr::extension::simple_op::MakeExtensionOp;
use hugr::ops::OpType;
use hugr::{Hugr, HugrView, Node};
use hugr_core::hugr::internal::HugrInternals;
use petgraph::Graph;

/// A call graph edge that contains a modifier.
pub struct ModifierCallGraphEdge<N = Node> {
    modifier: Option<Modifier>,
    edge: CallGraphEdge<N>,
}

impl<N> ModifierCallGraphEdge<N> {
    /// Create a new `ModifierCallGraphEdge` with the given modifier and edge.
    pub fn new(modifier: Option<Modifier>, edge: CallGraphEdge<N>) -> Self {
        Self { modifier, edge }
    }

    /// The modifier of this edge.
    pub fn modifier(self) -> Option<Modifier> {
        self.modifier
    }

    /// The function called by this edge.
    pub fn edge(&self) -> &CallGraphEdge<N> {
        &self.edge
    }
}

/// A call graph that contains modifiers.
pub struct ModifierCallGraph<T: HugrView> {
    g: Graph<
        CallGraphNode<<T as HugrInternals>::Node>,
        ModifierCallGraphEdge<<T as HugrInternals>::Node>,
    >,
    node_to_g: HashMap<<T as HugrInternals>::Node, petgraph::graph::NodeIndex<u32>>,
}

impl<T: HugrView> ModifierCallGraph<T> {
    pub fn new(circ: &Circuit<T>) -> Self
    where
        T: HugrView,
    {
        let mut g = Graph::default();
        let hugr = circ.hugr();
        let mut node_to_g = hugr
            .children(hugr.module_root())
            .filter_map(|n| {
                let weight = match hugr.get_optype(n) {
                    OpType::FuncDecl(_) => CallGraphNode::FuncDecl(n),
                    OpType::FuncDefn(_) => CallGraphNode::FuncDefn(n),
                    _ => return None,
                };
                Some((n, g.add_node(weight)))
            })
            .collect::<HashMap<_, _>>();
        if !hugr.entrypoint_optype().is_module() && !node_to_g.contains_key(&hugr.entrypoint()) {
            node_to_g.insert(hugr.entrypoint(), g.add_node(CallGraphNode::NonFuncRoot));
        }
        for (func, cg_node) in &node_to_g {
            Self::traverse(circ, *cg_node, *func, &mut g, &node_to_g);
        }
        ModifierCallGraph { g, node_to_g }
    }

    fn traverse(
        circ: &Circuit<T>,
        enclosing_func: petgraph::graph::NodeIndex<u32>,
        node: <T as HugrInternals>::Node,
        g: &mut Graph<
            CallGraphNode<<T as HugrInternals>::Node>,
            ModifierCallGraphEdge<<T as HugrInternals>::Node>,
        >,
        node_to_g: &HashMap<<T as HugrInternals>::Node, petgraph::graph::NodeIndex<u32>>,
    ) {
        let h = circ.hugr();
        for ch in h.children(node) {
            if h.get_optype(ch).is_func_defn() {
                continue;
            }
            Self::traverse(circ, enclosing_func, ch, g, node_to_g);
            let function = match h.get_optype(ch) {
                OpType::Call(_) => CallGraphEdge::Call(ch),
                OpType::LoadFunction(_) => CallGraphEdge::LoadFunction(ch),
                _ => continue,
            };
            // Determine the modifer by looking at the first child of the `LoadFunction`.
            let modifier = if let CallGraphEdge::LoadFunction(_) = function {
                h.first_child(ch)
                    .and_then(|n| Modifier::from_optype(circ.hugr().get_optype(n)))
            } else {
                None
            };
            let weight = ModifierCallGraphEdge::new(modifier, function);
            if let Some(target) = h.static_source(ch) {
                g.add_edge(enclosing_func, *node_to_g.get(&target).unwrap(), weight);
            }
        }
    }

    /// Allows access to the petgraph
    #[must_use]
    pub fn graph(
        &self,
    ) -> &Graph<
        CallGraphNode<<T as HugrInternals>::Node>,
        ModifierCallGraphEdge<<T as HugrInternals>::Node>,
    > {
        &self.g
    }

    /// Convert a Hugr [Node] into a petgraph node index.
    /// Result will be `None` if `n` is not a [`FuncDefn`](OpType::FuncDefn),
    /// [`FuncDecl`](OpType::FuncDecl) or the [HugrView::entrypoint].
    pub fn node_index(
        &self,
        n: <T as HugrInternals>::Node,
    ) -> Option<petgraph::graph::NodeIndex<u32>> {
        self.node_to_g.get(&n).copied()
    }
}
