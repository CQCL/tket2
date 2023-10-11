//! Distributed workers for the taso optimiser.

use std::thread::{self, JoinHandle};

use hugr::Hugr;

use crate::circuit::cost::CircuitCost;
use crate::circuit::CircuitHash;
use crate::rewrite::strategy::RewriteStrategy;
use crate::rewrite::Rewriter;

use super::hugr_pchannel::{PriorityChannelCommunication, Work};

/// A worker that processes circuits for the TASO optimiser.
pub struct TasoWorker<R, S, C, P: Ord> {
    /// The worker ID.
    #[allow(unused)]
    id: usize,
    /// The channel to send and receive work from.
    priority_channel: PriorityChannelCommunication<P>,
    /// The rewriter to use.
    rewriter: R,
    /// The rewrite strategy to use.
    strategy: S,
    /// The cost function
    cost_fn: C,
}

impl<R, S, C, P> TasoWorker<R, S, C, P>
where
    R: Rewriter + Send + 'static,
    S: RewriteStrategy + Send + 'static,
    C: Fn(&Hugr) -> P + Send + Sync + 'static,
    P: CircuitCost + Send + Sync + 'static,
{
    /// Spawn a new worker thread.
    #[allow(clippy::too_many_arguments)]
    pub fn spawn(
        id: usize,
        priority_channel: PriorityChannelCommunication<P>,
        rewriter: R,
        strategy: S,
        cost_fn: C,
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
                    cost_fn,
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
            let Ok(Work { circ, .. }) = self.priority_channel.recv() else {
                break;
            };

            let rewrites = self.rewriter.get_rewrites(&circ);
            let circs = self.strategy.apply_rewrites(rewrites, &circ);
            let new_circs = circs
                .into_iter()
                .map(|c| {
                    let hash = c.circuit_hash();
                    let cost = (self.cost_fn)(&c);
                    Work {
                        cost,
                        hash,
                        circ: c,
                    }
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
