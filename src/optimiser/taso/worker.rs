//! Distributed workers for the taso optimiser.

use std::thread::{self, JoinHandle};

use hugr::Hugr;

use crate::circuit::CircuitHash;
use crate::rewrite::strategy::RewriteStrategy;
use crate::rewrite::Rewriter;

pub fn spawn_pattern_matching_thread(
    rx_work: crossbeam_channel::Receiver<(u64, Hugr)>,
    tx_result: crossbeam_channel::Sender<Vec<(u64, Hugr)>>,
    rewriter: impl Rewriter + Send + 'static,
    strategy: impl RewriteStrategy + Send + 'static,
) -> JoinHandle<()> {
    thread::spawn(move || {
        // Process work until the main thread closes the channel send or receive
        // channel.
        while let Ok((_hash, circ)) = rx_work.recv() {
            let rewrites = rewriter.get_rewrites(&circ);
            let circs = strategy.apply_rewrites(rewrites, &circ);
            let hashed_circs = circs.into_iter().map(|c| (c.circuit_hash(), c)).collect();
            let send = tx_result.send(hashed_circs);
            if send.is_err() {
                // The main thread closed the send channel, we can stop.
                break;
            }
        }
    })
}
