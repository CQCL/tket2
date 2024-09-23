//! Distributed workers for the badger optimiser.

use std::thread::{self, JoinHandle};

use crate::circuit::CircuitHash;
use crate::rewrite::strategy::StrategyCost;
use crate::rewrite::Rewriter;
use crate::static_circ::UpdatableHash;
use crate::{circuit::cost::CircuitCost, Circuit};

use super::hugr_pchannel::{PriorityChannelCommunication, Work};

/// A worker that processes circuits for the Badger optimiser.
pub struct BadgerWorker<R, Cost, P: Ord> {
    /// The worker ID.
    #[allow(unused)]
    id: usize,
    /// The channel to send and receive work from.
    priority_channel: PriorityChannelCommunication<P>,
    /// The rewriter to use.
    rewriter: R,
    /// The rewrite strategy to use.
    cost: Cost,
}

impl<R, Cost, P> BadgerWorker<R, Cost, P>
where
    R: Rewriter<Circuit, Cost> + Send + 'static,
    Cost: StrategyCost + Send + 'static,
    P: CircuitCost + Send + Sync + 'static,
{
    /// Spawn a new worker thread.
    #[allow(clippy::too_many_arguments)]
    pub fn spawn(
        id: usize,
        priority_channel: PriorityChannelCommunication<P>,
        rewriter: R,
        strategy: Cost,
    ) -> JoinHandle<()> {
        let name = format!("BadgerWorker-{id}");
        thread::Builder::new()
            .name(name)
            .spawn(move || {
                let mut worker = Self {
                    id,
                    priority_channel,
                    rewriter,
                    cost: strategy,
                };
                worker.run_loop()
            })
            .unwrap()
    }

    /// Main loop of the worker.
    ///
    /// Processes work until the main thread closes the channel send or receive
    /// channel.
    #[tracing::instrument(target = "badger::metrics", skip(self))]
    fn run_loop(&mut self) {
        loop {
            let Ok(Work { circ, cost, .. }) = self.priority_channel.recv() else {
                break;
            };

            let rewrites = self.rewriter.get_rewrites(&circ);
            let max_cost = self.priority_channel.max_cost();
            let fast_hasher = UpdatableHash::with_static(&circ);
            let new_circs = rewrites
                .into_iter()
                .filter_map(|r| {
                    let new_cost = cost.add_delta(&r.cost_delta);
                    if max_cost.is_some() && &new_cost >= max_cost.as_ref().unwrap() {
                        return None;
                    }

                    let Ok(hash) = r.circ.circuit_hash() else {
                        // The composed rewrites were not valid.
                        //
                        // See [https://github.com/CQCL/tket2/discussions/242]
                        return None;
                    };

                    Some(Work {
                        cost: new_cost,
                        hash,
                        circ: r.circ,
                    })
                })
                .collect();

            let send = tracing::trace_span!(target: "badger::metrics", "BadgerWorker::send_result")
                .in_scope(|| self.priority_channel.send(new_circs));
            if send.is_err() {
                // Terminating
                break;
            }
        }
    }
}
