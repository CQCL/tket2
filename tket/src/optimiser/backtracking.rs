//! A backtracking optimiser.
//!
//! Will greedily explore the best states in the search space, and backtrack
//! when a state is found that is worse than the best state in the queue.

use std::time::Instant;

use super::{badger::BadgerOptions, pqueue::Entry, BadgerLogger, Optimiser, State, StatePQueue};

/// A single-threaded backtracking optimiser.
///
/// Will greedily explore the best states in the search space, and backtrack
/// when a state is found that is worse than the best state in the queue.
#[derive(Copy, Clone, Debug)]
pub struct BacktrackingOptimiser {
    /// The maximum size of the states priority queue.
    ///
    /// Defaults to `20`.
    pub queue_size: usize,
    /// The maximum time (in seconds) to run the optimiser.
    ///
    /// Defaults to `None`, which means no timeout.
    pub timeout: Option<u64>,
    /// The maximum time (in seconds) to search for new improvements to the
    /// best state. If no progress is made in this time, the optimiser will
    /// stop.
    ///
    /// Defaults to `None`, which means no timeout.
    pub progress_timeout: Option<u64>,
    /// The maximum number of states to process before stopping the
    /// optimisation.
    ///
    /// Defaults to `None`, which means no limit.
    pub max_visited_count: Option<usize>,
}

impl Default for BacktrackingOptimiser {
    fn default() -> Self {
        Self {
            queue_size: 20,
            timeout: None,
            progress_timeout: None,
            max_visited_count: None,
        }
    }
}

impl BacktrackingOptimiser {
    pub(super) fn with_badger_options(options: &BadgerOptions) -> Self {
        Self {
            queue_size: options.queue_size,
            timeout: options.timeout,
            progress_timeout: options.progress_timeout,
            max_visited_count: options.max_circuit_count,
        }
    }
}

impl Optimiser for BacktrackingOptimiser {
    fn optimise_with_log<C, S>(
        &self,
        start_state: S,
        mut context: C,
        mut logger: BadgerLogger,
    ) -> Option<S>
    where
        S: State<C>,
    {
        let start_time = Instant::now();
        let mut last_best_time = Instant::now();

        let mut best_state = start_state.clone();
        let mut best_cost = best_state.cost(&context)?;
        logger.log_best(&best_cost, None);

        // Priority queue of states to be processed
        let mut pq = StatePQueue::new(self.queue_size);
        pq.push(start_state, &context)?;

        let mut visited_count = 0;
        let mut timeout_flag = false;
        while let Some(Entry { state, cost, .. }) = pq.pop() {
            if cost < best_cost {
                best_state = state.clone();
                best_cost = cost.clone();
                // let num_rewrites = best_state.rewrite_trace().map(|rs| rs.count());
                // TODO: retrieve num_rewrites from context
                logger.log_best(&best_cost, None);
                last_best_time = Instant::now();
            }
            visited_count += 1;

            let new_states = state.next_states(&mut context);
            logger.register_branching_factor(new_states.len());

            for new_state in new_states {
                if pq.push(new_state, &context).is_some() {
                    logger.log_progress(visited_count, Some(pq.len()), pq.num_seen_hashes());
                }
            }

            if let Some(timeout) = self.timeout {
                if start_time.elapsed().as_secs() > timeout {
                    timeout_flag = true;
                    break;
                }
            }
            if let Some(p_timeout) = self.progress_timeout {
                if last_best_time.elapsed().as_secs() > p_timeout {
                    timeout_flag = true;
                    break;
                }
            }
            if let Some(max_visited_count) = self.max_visited_count {
                if visited_count >= max_visited_count {
                    timeout_flag = true;
                    break;
                }
            }
        }

        logger.log_processing_end(
            visited_count,
            Some(pq.num_seen_hashes()),
            best_cost,
            false,
            timeout_flag,
            start_time.elapsed(),
        );

        Some(best_state)
    }
}
