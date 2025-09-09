//! Badger circuit optimiser.
//!
//! This module implements the Badger circuit optimiser. It relies on a rewriter
//! and a RewriteStrategy instance to repeatedly rewrite a circuit and
//! optimising it according to some cost metric (typically gate count).
//!
//! The optimiser is implemented as a priority queue of circuits to be
//! processed. On top of the queue are the circuits with the lowest cost. They
//! are popped from the queue and replaced by the new circuits obtained from the
//! rewriter and the rewrite strategy. A hash of every circuit computed is
//! stored to detect and ignore duplicates. The priority queue is truncated
//! whenever it gets too large.

use hugr::hugr::views::sibling_subgraph::InvalidSubgraph;
use hugr::persistent::{Commit, PersistentHugr};
use hugr::{HugrView, Node};

use crate::circuit::cost::CircuitCost;
use crate::circuit::CircuitHash;
use crate::optimiser::{BacktrackingOptimiser, Optimiser, State};
use crate::resource::ResourceScope;
use crate::rewrite::strategy::RewriteStrategy;
use crate::rewrite::Rewriter;
use crate::rewrite_space::{CommittedRewrite, RewriteSpace};
use crate::Circuit;

/// Configuration options for the Badger optimiser.
#[derive(Copy, Clone, Debug)]
pub struct SeadogOptions {
    /// The maximum time (in seconds) to run the optimiser.
    ///
    /// Defaults to `None`, which means no timeout.
    pub timeout: Option<u64>,
    /// The maximum time (in seconds) to search for new improvements to the
    /// circuit. If no progress is made in this time, the optimiser will stop.
    ///
    /// Defaults to `None`, which means no timeout.
    pub progress_timeout: Option<u64>,
    /// The maximum number of circuits to process before stopping the
    /// optimisation.
    ///
    /// For data parallel multi-threading, (split_circuit=true), applies on a
    /// per-thread basis, otherwise applies globally.
    ///
    /// Defaults to `None`, which means no limit.
    pub max_circuit_count: Option<usize>,

    // pub n_threads: NonZeroUsize,
    // pub split_circuit: bool,
    /// The maximum size of the circuit candidates priority queue.
    ///
    /// Defaults to `20`.
    pub queue_size: usize,
}

impl Default for SeadogOptions {
    fn default() -> Self {
        Self {
            timeout: Default::default(),
            progress_timeout: Default::default(),
            queue_size: 20,
            max_circuit_count: None,
        }
    }
}

/// The Seadog optimiser.
#[derive(Clone, Debug)]
pub struct SeadogOptimiser<R, S> {
    rewriter: R,
    strategy: S,
}

/// A trait for rewriters that can be used with the Badger optimiser.
pub trait SeadogRewriter<C: 'static + CircuitCost>:
    for<'c> Rewriter<RewriteSpace<Cost<C>>, Rewrite<'c> = Commit<'c>> + Send + Clone + Sync + 'static
{
}

/// A trait for rewrite strategies that can be used with the Badger optimiser.
pub trait SeadogRewriteStrategy: RewriteStrategy + Send + Sync + Clone + 'static {}

impl<S> SeadogRewriteStrategy for S where S: RewriteStrategy + Send + Sync + Clone + 'static {}

impl<R, C: CircuitCost + 'static> SeadogRewriter<C> for R where
    R: for<'c> Rewriter<RewriteSpace<Cost<C>>, Rewrite<'c> = Commit<'c>>
        + Send
        + Clone
        + Sync
        + 'static
{
}

impl<R, S> SeadogOptimiser<R, S> {
    /// Create a new Badger optimiser.
    pub fn new(rewriter: R, strategy: S) -> Self {
        Self { rewriter, strategy }
    }

    fn cost<C: CircuitCost + 'static>(
        &self,
        circ: &ResourceScope<impl HugrView<Node = Node>>,
    ) -> S::Cost
    where
        R: SeadogRewriter<C>,
        S: SeadogRewriteStrategy<Cost = C>,
    {
        self.strategy.circuit_cost(circ)
    }
}

/// The cost of a rewrite in seadog.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, serde::Serialize)]
pub struct Cost<C: CircuitCost> {
    n_rewrites_since_salient: usize,
    rewrite_cost: C::CostDelta,
    total_cost: C,
}

/// A state in the Badger search space.
#[derive(Clone, Debug)]
struct SeadogState<'c> {
    /// The commit of the current state.
    commit: CommittedRewrite<'c>,
}

struct SeadogContext<'o, R, S: SeadogRewriteStrategy> {
    rewrite_space: &'o RewriteSpace<Cost<S::Cost>>,
    optimiser: &'o SeadogOptimiser<R, S>,
}

impl<'o, R, S> State<SeadogContext<'o, R, S>> for SeadogState<'o>
where
    S::Cost: serde::Serialize + 'static,
    <S::Cost as CircuitCost>::CostDelta: serde::Serialize + 'static,
    R: SeadogRewriter<S::Cost>,
    S: SeadogRewriteStrategy,
{
    type Cost = Cost<S::Cost>;

    fn hash(&self, _context: &SeadogContext<'o, R, S>) -> Option<u64> {
        let circ = Circuit::new(PersistentHugr::from_commit(self.commit.clone().into()));
        circ.circuit_hash(circ.parent()).ok()
    }

    fn cost(&self, context: &SeadogContext<'o, R, S>) -> Option<Self::Cost> {
        let cost = context.rewrite_space.get_cost(&self.commit);
        Some(cost.clone())
    }

    fn next_states(&self, context: &mut SeadogContext<'o, R, S>) -> Vec<Self> {
        let commit: Commit = self.commit.clone().into();
        let inserted_nodes = commit.inserted_nodes().map(|n| commit.to_patch_node(n));
        let phugr = PersistentHugr::from_commit(commit.clone());

        // TODO: expand inserted_nodes to some radius
        let rewrites = inserted_nodes.flat_map(|n| {
            context
                .optimiser
                .rewriter
                .get_rewrites(&context.rewrite_space, n)
        });

        let mut commited_rewrites = Vec::with_capacity(rewrites.size_hint().0);
        for rw in rewrites {
            let old_cost = self.cost(context).unwrap();
            let new_nodes_cost: S::Cost = rw
                .inserted_nodes()
                .map(|n| context.optimiser.strategy.op_cost(rw.get_optype(n)))
                .sum();
            let delta = new_nodes_cost.sub_cost(
                &rw.deleted_parent_nodes()
                    .map(|n| context.optimiser.strategy.op_cost(phugr.get_optype(n)))
                    .sum(),
            );

            let new_total_cost = old_cost.total_cost.add_delta(&delta);
            let new_n_rewrites_since_salient = if is_salient(&new_total_cost, &old_cost.total_cost)
            {
                0
            } else {
                old_cost.n_rewrites_since_salient + 1
            };

            let commit = context.rewrite_space.add_from_commit(
                rw,
                Cost {
                    n_rewrites_since_salient: new_n_rewrites_since_salient,
                    rewrite_cost: delta,
                    total_cost: new_total_cost,
                },
            );
            commited_rewrites.push(SeadogState { commit });
        }

        commited_rewrites
    }
}

fn is_salient<C>(new_total_cost: &C, total_cost: &C) -> bool
where
    C: CircuitCost + serde::Serialize + 'static,
{
    new_total_cost.as_usize() < total_cost.as_usize()
}

impl<R, S> SeadogOptimiser<R, S>
where
    R: SeadogRewriter<S::Cost>,
    S: SeadogRewriteStrategy,
    S::Cost: serde::Serialize + Send + Sync,
    <S::Cost as CircuitCost>::CostDelta: serde::Serialize + 'static,
{
    /// Run the Badger optimiser on a circuit.
    ///
    /// A timeout (in seconds) can be provided.
    pub fn optimise(
        &self,
        circ: &Circuit<impl HugrView<Node = Node>>,
        options: SeadogOptions,
    ) -> Circuit {
        let h = self.seadog(circ, options).to_hugr();
        Circuit::new(h)
    }

    /// Run the Seadog optimiser on a circuit, using a single thread.
    fn seadog(
        &self,
        circ: &Circuit<impl HugrView<Node = Node>>,
        opt: SeadogOptions,
    ) -> PersistentHugr {
        let circ = circ.to_owned();

        if circ.try_to_subgraph() == Err(InvalidSubgraph::EmptySubgraph) {
            // No rewrites possible in an empty circuit
            panic!("Empty circuit input not supported; no optimisation possible");
        }

        let backtracking = BacktrackingOptimiser::with_seadog_options(&opt);
        let circ = ResourceScope::from_circuit(circ);
        let cost = self.cost(&circ);

        let rewrite_space = RewriteSpace::new();
        let init_commit = rewrite_space
            .try_set_base(
                circ.into_hugr(),
                Cost {
                    n_rewrites_since_salient: 0,
                    rewrite_cost: <S::Cost as CircuitCost>::CostDelta::default(),
                    total_cost: cost,
                },
            )
            .unwrap();

        let init_state = SeadogState {
            commit: init_commit,
        };

        let _opt_state = backtracking
            .optimise(
                init_state,
                SeadogContext {
                    rewrite_space: &rewrite_space,
                    optimiser: self,
                },
            )
            .expect("optimisation failed");

        rewrite_space
            .extract_best_with_cost(|c| c.rewrite_cost.clone())
            .unwrap()
    }
}
