//! A multi-producer multi-consumer min-priority channel of Hugrs.

use std::thread;

use crossbeam_channel::{select, Receiver, Sender};
use fxhash::FxHashSet;
use hugr::Hugr;

use super::hugr_pqueue::{Entry, HugrPQ};

/// A priority channel for HUGRs.
///
/// Queues hugrs using a cost function `C` that produces priority values `P`.
///
/// Uses a thread internally to orchestrate the queueing.
pub(super) struct HugrPriorityChannel<C, P: Ord> {
    // Channels to add and remove circuits from the queue.
    push: Receiver<Vec<(u64, Hugr)>>,
    pop: Sender<(u64, Hugr)>,
    // Outbound channel to log to main thread.
    log: Sender<PriorityChannelLog<P>>,
    // Inbound channel to be terminated.
    timeout: Receiver<()>,
    // The priority queue data structure.
    pq: HugrPQ<P, C>,
    // The set of hashes we've seen.
    seen_hashes: FxHashSet<u64>,
    // The minimum cost we've seen.
    min_cost: Option<P>,
    // The number of circuits we've seen (for logging).
    circ_cnt: usize,
}

pub(super) type Item = (u64, Hugr);

/// Logging information from the priority channel.
pub(super) enum PriorityChannelLog<C> {
    NewBestCircuit(Hugr, C),
    CircuitCount(usize, usize),
}

/// Channels for communication with the priority channel.
pub(super) struct PriorityChannelCommunication<C> {
    pub(super) push: Option<Sender<Vec<Item>>>,
    pub(super) pop: Option<Receiver<Item>>,
    pub(super) log: Receiver<PriorityChannelLog<C>>,
    timeout: Sender<()>,
}

impl<C> PriorityChannelCommunication<C> {
    /// Send Timeout signal to the priority channel.
    pub(super) fn timeout(&self) {
        self.timeout.send(()).unwrap();
    }

    /// Close the local copies of the push and pop channels.
    pub(super) fn drop_pop_push(&mut self) {
        self.pop = None;
        self.push = None;
    }
}

impl<C, P> HugrPriorityChannel<C, P>
where
    C: Fn(&Hugr) -> P + Send + Sync + 'static,
    P: Ord + Send + Sync + Clone + 'static,
{
    /// Initialize the queueing system.
    ///
    /// Start the Hugr priority queue in a new thread.
    ///
    /// Get back channels for communication with the priority queue
    ///  - push/pop channels for adding and removing circuits to/from the queue,
    ///  - a channel on which to receive logging information, and
    ///  - a channel on which to send a timeout signal.
    pub(super) fn init(cost_fn: C, queue_capacity: usize) -> PriorityChannelCommunication<P> {
        // channels for pushing and popping circuits from pqueue
        let (tx_push, rx_push) = crossbeam_channel::unbounded();
        let (tx_pop, rx_pop) = crossbeam_channel::bounded(0);
        // channels for communication with main (logging, minimum circuits and timeout)
        let (tx_log, rx_log) = crossbeam_channel::unbounded();
        let (tx_timeout, rx_timeout) = crossbeam_channel::bounded(0);
        let pq =
            HugrPriorityChannel::new(rx_push, tx_pop, tx_log, rx_timeout, cost_fn, queue_capacity);
        pq.run();
        PriorityChannelCommunication {
            push: Some(tx_push),
            pop: Some(rx_pop),
            log: rx_log,
            timeout: tx_timeout,
        }
    }

    fn new(
        push: Receiver<Vec<(u64, Hugr)>>,
        pop: Sender<(u64, Hugr)>,
        log: Sender<PriorityChannelLog<P>>,
        timeout: Receiver<()>,
        cost_fn: C,
        queue_capacity: usize,
    ) -> Self {
        // The priority queue, local to this thread.
        let pq = HugrPQ::new(cost_fn, queue_capacity);
        // The set of hashes we've seen.
        let seen_hashes = FxHashSet::default();
        // The minimum cost we've seen.
        let min_cost = None;
        // The number of circuits we've seen (for logging).
        let circ_cnt = 0;

        HugrPriorityChannel {
            push,
            pop,
            log,
            timeout,
            pq,
            seen_hashes,
            min_cost,
            circ_cnt,
        }
    }

    /// Run the queuer as a thread.
    fn run(mut self) {
        let builder = thread::Builder::new().name("priority queueing".into());
        let _ = builder
            .name("priority-channel".into())
            .spawn(move || {
                loop {
                    if self.pq.is_empty() {
                        let Ok(new_circs) = self.push.recv() else {
                            // The senders have closed the channel, we can stop.
                            break;
                        };
                        self.enqueue_circs(new_circs);
                    } else {
                        select! {
                            recv(self.push) -> result => {
                                let Ok(new_circs) = result else {
                                    // The senders have closed the channel, we can stop.
                                    break;
                                };
                                self.enqueue_circs(new_circs);
                            }
                            send(self.pop, {let Entry {hash, circ, ..} = self.pq.pop().unwrap(); (hash, circ)}) -> result => {
                                match result {
                                    Ok(()) => {},
                                    // The receivers have closed the channel, we can stop.
                                    Err(_) => break,
                                }
                            }
                            recv(self.timeout) -> _ => {
                                // We've timed out.
                                break
                            }
                        }
                    }
                }
                // Send a last set of logs before terminating.
                self.log
                    .send(PriorityChannelLog::CircuitCount(
                        self.circ_cnt,
                        self.seen_hashes.len(),
                    ))
                    .unwrap();
            })
            .unwrap();
    }

    /// Add circuits to queue.
    #[tracing::instrument(target = "taso::metrics", skip(self, circs))]
    fn enqueue_circs(&mut self, circs: Vec<(u64, Hugr)>) {
        for (hash, circ) in circs {
            let cost = (self.pq.cost_fn)(&circ);
            if !self.seen_hashes.insert(hash) {
                // Ignore this circuit: we've seen it before.
                continue;
            }

            // A new best circuit
            if self.min_cost.is_none() || Some(&cost) < self.min_cost.as_ref() {
                self.min_cost = Some(cost.clone());
                self.log
                    .send(PriorityChannelLog::NewBestCircuit(
                        circ.clone(),
                        cost.clone(),
                    ))
                    .unwrap();
            }

            self.circ_cnt += 1;
            self.pq.push_unchecked(circ, hash, cost);

            // Send logs every 1000 circuits.
            if self.circ_cnt % 1000 == 0 {
                // TODO: Add a minimum time between logs
                self.log
                    .send(PriorityChannelLog::CircuitCount(
                        self.circ_cnt,
                        self.seen_hashes.len(),
                    ))
                    .unwrap();
            }
        }
    }
}
