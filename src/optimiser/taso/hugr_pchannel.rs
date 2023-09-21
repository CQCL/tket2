//! A multi-producer multi-consumer min-priority channel of Hugrs.

use std::marker::PhantomData;
use std::thread;

use crossbeam_channel::{select, Receiver, Sender};
use hugr::Hugr;

use super::hugr_pqueue::{Entry, HugrPQ};

/// A priority channel for HUGRs.
///
/// Queues hugrs using a cost function `C` that produces priority values `P`.
///
/// Uses a thread internally to orchestrate the queueing.
pub struct HugrPriorityChannel<P, C> {
    _phantom: PhantomData<(P, C)>,
}

pub type Item = (u64, Hugr);

impl<P: Ord, C> HugrPriorityChannel<P, C>
where
    C: Fn(&Hugr) -> P + Send + Sync + 'static,
{
    /// Initialize the queueing system.
    ///
    /// Get back a channel on which to queue hugrs with their hash, and
    /// a channel on which to receive the output.    
    pub fn init(cost_fn: C, queue_capacity: usize) -> (Sender<Vec<Item>>, Receiver<Item>) {
        let (ins, inr) = crossbeam_channel::unbounded();
        let (outs, outr) = crossbeam_channel::bounded(0);
        Self::run(inr, outs, cost_fn, queue_capacity);
        (ins, outr)
    }

    /// Run the queuer as a thread.
    fn run(
        in_channel_orig: Receiver<Vec<(u64, Hugr)>>,
        out_channel_orig: Sender<(u64, Hugr)>,
        cost_fn: C,
        queue_capacity: usize,
    ) {
        let builder = thread::Builder::new().name("priority queueing".into());
        let in_channel = in_channel_orig.clone();
        let out_channel = out_channel_orig.clone();
        let _ = builder
            .spawn(move || {
                // The priority queue, local to this thread.
                let mut pq: HugrPQ<P, C> =
                    HugrPQ::with_capacity(cost_fn, queue_capacity);

                loop {
                    if pq.is_empty() {
                        // Nothing queued to go out. Wait for input.
                        match in_channel.recv() {
                            Ok(new_circs) => {
                                for (hash, circ) in new_circs {
                                    pq.push_with_hash_unchecked(circ, hash);
                                }
                            }
                            // The sender has closed the channel, we can stop.
                            Err(_) => break,
                        }
                    } else {
                        select! {
                            recv(in_channel) -> result => {
                                match result {
                                    Ok(new_circs) => {
                                        for (hash, circ) in new_circs {
                                            pq.push_with_hash_unchecked(circ, hash);
                                        }
                                    }
                                    // The sender has closed the channel, we can stop.
                                    Err(_) => break,
                                }
                            }
                            send(out_channel, {let Entry {hash, circ, ..} = pq.pop().unwrap(); (hash, circ)}) -> result => {
                                match result {
                                    Ok(()) => {},
                                    // The receivers have closed the channel, we can stop.
                                    Err(_) => break,
                                }
                            }
                        }
                    }
                    if pq.len() >= queue_capacity {
                        pq.truncate(queue_capacity / 2);
                    }
                }
            })
            .unwrap();
    }
}
