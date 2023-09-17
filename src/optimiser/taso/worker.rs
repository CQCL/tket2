//! Distributed workers for the taso optimiser.

use std::thread::{self, JoinHandle};

use hugr::Hugr;

use crate::circuit::CircuitHash;
use crate::rewrite::strategy::RewriteStrategy;
use crate::rewrite::Rewriter;

/// A worker that processes circuits for the TASO optimiser.
pub struct TasoWorker<R, S> {
    _phantom: std::marker::PhantomData<(R, S)>,
}

impl<R, S> TasoWorker<R, S>
where
    R: Rewriter + Send + 'static,
    S: RewriteStrategy + Send + 'static,
{
    /// Spawn a new worker thread.
    pub fn spawn(
        rx_work: crossbeam_channel::Receiver<(u64, Hugr)>,
        tx_result: crossbeam_channel::Sender<Vec<(u64, Hugr)>>,
        rewriter: R,
        strategy: S,
        worker_name: Option<String>,
    ) -> JoinHandle<()> {
        let mut builder = thread::Builder::new();
        if let Some(name) = worker_name {
            builder = builder.name(name);
        };
        builder
            .spawn(move || Self::worker_loop(rx_work, tx_result, rewriter, strategy))
            .unwrap()
    }

    /// Main loop of the worker.
    ///
    /// Processes work until the main thread closes the channel send or receive
    /// channel.
    #[tracing::instrument(target = "taso::metrics", skip_all)]
    fn worker_loop(
        rx_work: crossbeam_channel::Receiver<(u64, Hugr)>,
        tx_result: crossbeam_channel::Sender<Vec<(u64, Hugr)>>,
        rewriter: R,
        strategy: S,
    ) {
        while let Ok((_hash, circ)) = rx_work.recv() {
            let hashed_circs = Self::process_circ(circ, &rewriter, &strategy);
            let send = tracing::span!(tracing::Level::TRACE, "TasoWorker::send_result")
                .in_scope(|| tx_result.send(hashed_circs));
            if send.is_err() {
                // The main thread closed the send channel, we can stop.
                break;
            }
        }
    }

    /// Process a circuit.
    #[tracing::instrument(target = "taso::metrics", skip_all)]
    fn process_circ(circ: Hugr, rewriter: &R, strategy: &S) -> Vec<(u64, Hugr)> {
        let rewrites = rewriter.get_rewrites(&circ);
        let circs = strategy.apply_rewrites(rewrites, &circ);
        circs.into_iter().map(|c| (c.circuit_hash(), c)).collect()
    }
}
