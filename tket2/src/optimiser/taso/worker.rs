//! Distributed workers for the taso optimiser.

use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use crate::circuit::cost::CircuitCost;
use crate::circuit::CircuitHash;
use crate::optimiser::taso::log::PROGRESS_TARGET;
use crate::rewrite::strategy::RewriteStrategy;
use crate::rewrite::Rewriter;

use super::hugr_pchannel::{PriorityChannelCommunication, Work};

/// A worker that processes circuits for the TASO optimiser.
pub struct TasoWorker<R, S, P: Ord> {
    /// The worker ID.
    #[allow(unused)]
    id: usize,
    /// The channel to send and receive work from.
    priority_channel: PriorityChannelCommunication<P>,
    /// The rewriter to use.
    rewriter: R,
    /// The rewrite strategy to use.
    strategy: S,
}

impl<R, S, P> TasoWorker<R, S, P>
where
    R: Rewriter + Send + 'static,
    S: RewriteStrategy<Cost = P> + Send + 'static,
    P: CircuitCost + Send + Sync + 'static,
{
    /// Spawn a new worker thread.
    #[allow(clippy::too_many_arguments)]
    pub fn spawn(
        id: usize,
        priority_channel: PriorityChannelCommunication<P>,
        rewriter: R,
        strategy: S,
    ) -> JoinHandle<()> {
        let name = format!("TasoWorker-{id}");
        thread::Builder::new()
            .name(name)
            .spawn(move || {
                let mut worker = Self {
                    id,
                    priority_channel,
                    rewriter,
                    strategy,
                };
                worker.run_loop()
            })
            .unwrap()
    }

    /// Main loop of the worker.
    ///
    /// Processes work until the main thread closes the channel send or receive
    /// channel.
    #[tracing::instrument(target = "taso::metrics", skip(self))]
    fn run_loop(&mut self) {
        loop {
            let Ok(Work { circ, cost, .. }) = self.priority_channel.recv() else {
                break;
            };

            let time_start = Instant::now();

            let rewrites = self.rewriter.get_rewrites(&circ);
            let time_pmatch = Instant::now();

            let rewrite_result = self.strategy.apply_rewrites(rewrites, &circ);
            let max_cost = self.priority_channel.max_cost();
            let time_rewrite = Instant::now();

            let num_circs = rewrite_result.len();
            let mut duration_hash = Duration::default();
            let mut duration_cost = Duration::default();
            let new_circs = rewrite_result
                .into_iter()
                .filter_map(|(c, cost_delta)| {
                    let time_circ = Instant::now();

                    let new_cost = cost.add_delta(&cost_delta);
                    let time_cost = Instant::now();
                    duration_cost += time_cost - time_circ;
                    if max_cost.is_some() && &new_cost >= max_cost.as_ref().unwrap() {
                        return None;
                    }

                    let hash = c.circuit_hash();
                    let time_hash = Instant::now();
                    duration_hash += time_hash - time_cost;

                    Some(Work {
                        cost: new_cost,
                        hash,
                        circ: c,
                    })
                })
                .collect();
            let time_process = Instant::now();

            let send = tracing::trace_span!(target: "taso::metrics", "TasoWorker::send_result")
                .in_scope(|| self.priority_channel.send(new_circs));
            if send.is_err() {
                // Terminating
                break;
            }
            let time_send = Instant::now();

            tracing::info!(
                target: PROGRESS_TARGET,
                "Processed circuit ({num_circs} matches). matching {:4}ms    rewriting {:4}ms    hashing {:4}ms    cost {:4}ms    enqueuing {:4}ms",
                (time_pmatch - time_start).as_millis(),
                (time_rewrite - time_pmatch).as_millis(),
                duration_hash.as_millis(),
                duration_cost.as_millis(),
                (time_process - time_send).as_millis(),
            );
        }
    }
}
