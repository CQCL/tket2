//! Optimisers for circuit rewriting.
//!
//! Currently, the only optimiser is Badger

mod backtracking;
pub mod badger;
mod pqueue;
mod pqueue_worker;

use std::fmt::Debug;

#[cfg(feature = "portmatching")]
pub use badger::ECCBadgerOptimiser;
pub use badger::{BadgerLogger, BadgerOptimiser};
use pqueue::StatePQueue;
use pqueue_worker::StatePQWorker;

pub use backtracking::BacktrackingOptimiser;

/// An optimiser exploring a discrete search space, in search for the lowest
/// cost state.
///
/// The optimiser accepts a global context object, which each state can mutate
/// when computing the next states that can be transitioned to.
pub trait Optimiser: Sized {
    /// Start optimisation from the given state, using the given context.
    fn optimise<C, S>(&self, start_state: S, start_context: C) -> Option<S>
    where
        S: State<C>,
    {
        self.optimise_with_log(start_state, start_context, Default::default())
    }

    /// Start optimisation from the given state, using the given context and
    /// logger.
    fn optimise_with_log<C, S>(
        &self,
        start_state: S,
        start_context: C,
        log: BadgerLogger,
    ) -> Option<S>
    where
        S: State<C>;
}

/// A state in the search space of the optimiser.
///
/// A mutable context is shared between all states in the search space.
pub trait State<Context>: Clone {
    /// The cost of the state, to be minimised.
    type Cost: Ord + Debug + serde::Serialize + Clone;

    /// The hash of the state.
    ///
    /// This may fail, in which case the state is considered invalid and
    /// discarded.
    fn hash(&self, context: &Context) -> Option<u64>;

    /// The cost of the state, to be minimised.
    ///
    /// This may fail, in which case the state is considered invalid and
    /// discarded.
    fn cost(&self, context: &Context) -> Option<Self::Cost>;

    /// The next states from the current state.
    ///
    /// States are allowed to write to a global "context" object. This allows
    /// search problems where for instance states are nodes in an infinitely
    /// sized graph that is lazily generated.
    ///
    /// Note that the order in which the states are visited is not guaranteed,
    /// which may result in different contexts.
    fn next_states(&self, context: &mut Context) -> Vec<Self>;
}
