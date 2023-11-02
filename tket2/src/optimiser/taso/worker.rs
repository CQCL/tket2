//! Distributed workers for the taso optimiser.

use std::thread::{self, JoinHandle};

use crate::circuit::cost::CircuitCost;
use crate::circuit::CircuitHash;
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

            let rewrites = self.rewriter.get_rewrites(&circ);
            let rewrite_result = self.strategy.apply_rewrites(rewrites, &circ);
            let max_cost = self.priority_channel.max_cost();
            let new_circs = rewrite_result
                .into_iter()
                .filter_map(|(c, cost_delta)| {
                    let new_cost = cost.add_delta(&cost_delta);
                    if max_cost.is_some() && &new_cost >= max_cost.as_ref().unwrap() {
                        return None;
                    }

                    let hash = c.circuit_hash();
                    Some(Work {
                        cost: new_cost,
                        hash,
                        circ: c,
                    })
                })
                .collect();

            let send = tracing::trace_span!(target: "taso::metrics", "TasoWorker::send_result")
                .in_scope(|| self.priority_channel.send(new_circs));
            if send.is_err() {
                // Terminating
                break;
            }
        }
    }
}
