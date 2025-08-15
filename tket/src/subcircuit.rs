//! Subcircuits of circuits.
//!
//! This provides a specialised version of hugr's [`SiblingSubgraph`], for the
//! case where the HUGR region is a [`Circuit`].

use hugr::hugr::views::sibling_subgraph::{InvalidReplacement, InvalidSubgraph};
use hugr::hugr::views::SiblingSubgraph;
use hugr::HugrView;
use itertools::Itertools;
use portgraph::algorithms::convex::{LineIndex, LineInterval, LineIntervals};

use crate::circuit::Circuit;
use crate::rewrite::CircuitRewrite;

pub use hugr::hugr::views::sibling_subgraph::LineConvexChecker;

/// A subcircuit of a circuit.
///
/// Store convex subgraphs of Circuit-like regions of HUGRs as intervals on the
/// "lines" of the circuit (see below and [`LineConvexChecker`]).
///
/// Lines are the result of a partition of the DFG
/// into edge-disjoint paths; in the case of purely quantum ops, the lines are
/// the qubits
///
/// ## Differences with [`SiblingSubgraph`]
///
/// [`Subcircuit`]s have two main distinguishing features over
/// [`SiblingSubgraph`], that make it more suitable in some contexts, especially
/// when working with subgraphs of circuits:
///  - The representation of the subcircuit uses the circuit structure to be more
///    memory efficient. The size of a [`Subcircuit`] instance is not linear in
///    the number of nodes in the subgraph, but linear in the number of lines
///    (~qubits) in the subcircuit.
///  - Subcircuits can be updated by extending the intervals without having to
///    recompute the subgraph from scratch and rechecking convexity.
///
/// Along with these features come two caveats to be aware of:
///  - Unlike subgraphs, [`Subcircuit`]s do not maintain an ordered boundary.
///    This is because the ordering would be hard to maintain when the subcircuit
///    is expanded. When a subcircuit is converted into a [`SiblingSubgraph`],
///    a boundary order gets fixed.
///  - Constructing a [`LineConvexChecker`] (required to construct [`Subcircuit`]s)
///    is currently ~2.5x more expensive than constructing the default
///    [`TopoConvexChecker`]. This must be amortised over many subcircuit
///    constructions.
///
/// [`TopoConvexChecker`]: hugr::hugr::views::sibling_subgraph::TopoConvexChecker
#[derive(Debug, Clone, PartialEq)]
pub struct Subcircuit {
    intervals: LineIntervals,
}

impl Subcircuit {
    /// Create a new subcircuit induced from a single node.
    ///
    /// This requires a [`LineConvexChecker`], which can be created from a
    /// circuit using [`LineConvexChecker::from_entrypoint`].
    #[inline(always)]
    pub fn from_node<H: HugrView>(node: H::Node, checker: &LineConvexChecker<H>) -> Self {
        Self::try_from_nodes([node], checker).expect("single node is a valid subcircuit")
    }

    /// Create a new subcircuit induced from a set of nodes.
    /// [`Circuit`].
    ///
    /// This requires a [`LineConvexChecker`], which can be created from a
    /// circuit using [`LineConvexChecker::from_entrypoint`].
    pub fn try_from_nodes<H: HugrView>(
        nodes: impl IntoIterator<Item = H::Node>,
        checker: &LineConvexChecker<H>,
    ) -> Result<Self, InvalidSubgraph<H::Node>> {
        let intervals = checker
            .get_intervals_from_nodes(nodes.into_iter())
            .ok_or(InvalidSubgraph::NotConvex)?;
        Ok(Self { intervals })
    }

    /// Create a new subcircuit from a set of intervals.
    ///
    /// Warning: this does not check that the intervals represent a valid
    /// subcircuit.
    pub fn from_intervals_unchecked(intervals: LineIntervals) -> Self {
        Self { intervals }
    }

    /// Create a new empty subcircuit.
    pub fn new_empty() -> Self {
        Self {
            intervals: LineIntervals::default(),
        }
    }

    /// Nodes in the subcircuit.
    pub fn nodes<'a, 'g: 'a, H: 'g + HugrView>(
        &'a self,
        checker: &'a LineConvexChecker<'g, H>,
    ) -> impl Iterator<Item = H::Node> + 'a + use<'a, 'g, H> {
        checker.nodes_in_intervals(&self.intervals)
    }

    /// Number of nodes in the subcircuit.
    pub fn node_count<'a, H: HugrView>(&'a self, checker: &'a LineConvexChecker<H>) -> usize {
        self.nodes(checker).count()
    }

    /// Whether the subcircuit is empty.
    pub fn is_empty(&self) -> bool {
        self.intervals.is_empty()
    }

    /// Get the interval for the given line.
    pub fn get_interval(&self, line: LineIndex) -> Option<LineInterval> {
        self.intervals.get(line)
    }

    /// Iterate over the line indices of the subcircuit and their intervals.
    pub fn intervals_iter(&self) -> impl Iterator<Item = (LineIndex, LineInterval)> + '_ {
        self.intervals.iter()
    }

    /// Convert the subcircuit to a [`SiblingSubgraph`].
    pub fn try_to_subgraph<H: HugrView>(
        &self,
        checker: &LineConvexChecker<H>,
    ) -> Result<SiblingSubgraph<H::Node>, InvalidSubgraph<H::Node>> {
        let Self { intervals } = self;
        SiblingSubgraph::try_from_nodes_with_intervals(
            checker.nodes_in_intervals(intervals).collect_vec(),
            intervals,
            checker,
        )
    }

    /// Create a rewrite rule to replace the subcircuit with a new circuit.
    ///
    /// # Parameters
    /// * `circuit` - The base circuit that contains the subcircuit.
    /// * `replacement` - The new circuit to replace the subcircuit with.
    pub fn create_rewrite(
        &self,
        replacement: Circuit<impl HugrView<Node = hugr::Node>>,
        checker: &LineConvexChecker<impl HugrView<Node = hugr::Node>>,
    ) -> Result<CircuitRewrite<hugr::Node>, InvalidReplacement> {
        // The replacement must be a Dfg rooted hugr.
        let hugr = checker.hugr();
        let subgraph = self
            .try_to_subgraph(checker)
            .map_err(|_| InvalidReplacement::NonConvexSubgraph)?;
        CircuitRewrite::try_new(&subgraph, hugr, replacement)
    }

    /// Extend the subcircuit to include the given node.
    ///
    /// Return whether the subcircuit was successfully extended to contain `node`,
    /// i.e. whether adding `node` to the subgraph represented by the intervals
    /// results in another subgraph that can be expressed as line intervals.
    ///
    /// If `false` is returned, `self` is left unchanged.
    pub fn try_extend<H: HugrView>(
        &mut self,
        node: H::Node,
        checker: &LineConvexChecker<H>,
    ) -> bool {
        checker.try_extend_intervals(&mut self.intervals, node)
    }
}
