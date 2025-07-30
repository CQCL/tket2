//! A multi-producer multi-consumer min-priority channel of Hugrs.

use std::sync::{Arc, RwLock};
use std::thread;
use std::time::Instant;

use crossbeam_channel::{select, Receiver, RecvError, SendError, Sender};
use fxhash::FxHashSet;

use crate::circuit::cost::CircuitCost;
use crate::Circuit;

use super::hugr_pqueue::{Entry, HugrPQ};

/// A unit of work for a worker, consisting of a circuit to process, along its
/// hash and cost.
pub type Work<P> = Entry<Circuit, P, u64>;

/// A priority channel for circuits.
///
/// Queues circuits using a cost function `C` that produces priority values `P`.
///
/// Uses a thread internally to orchestrate the queueing.
#[derive(Debug, Clone)]
pub struct HugrPriorityChannel<C, P: Ord> {
    /// Channel to add circuits from the queue.
    push: Receiver<Vec<Work<P>>>,
    /// Channel to pop circuits from the queue.
    pop: Sender<Work<P>>,
    /// Outbound channel to log to main thread.
    log: Sender<PriorityChannelLog<P>>,
    /// Timestamp of the last progress log.
    /// Used to avoid spamming the log.
    last_progress_log: Instant,
    /// The priority queue data structure.
    pq: HugrPQ<P, C>,
    /// The set of hashes we've seen.
    seen_hashes: FxHashSet<u64>,
    /// The minimum cost we've seen.
    min_cost: Option<P>,
    /// The number of circuits we've processed.
    circ_cnt: usize,
    /// The maximum cost in the queue. Shared with the workers so they can cull
    /// the circuits they generate.
    max_cost: Arc<RwLock<Option<P>>>,
    /// Local copy of `max_cost`, used to avoid locking when checking the value.
    local_max_cost: Option<P>,
}

/// Logging information from the priority channel.
#[derive(Debug, Clone)]
pub enum PriorityChannelLog<P> {
    NewBestCircuit(Circuit, P),
    CircuitCount {
        processed_count: usize,
        seen_count: usize,
        queue_length: usize,
    },
}

/// Channels for communication with the priority channel.
#[derive(Clone)]
pub struct PriorityChannelCommunication<P> {
    /// A channel to add batches of circuits to the queue.
    push: Sender<Vec<Work<P>>>,
    /// A channel to remove the best candidate circuit from the queue.
    pop: Receiver<Work<P>>,
    /// A maximum accepted cost for the queue. Circuits with higher costs will
    /// be dropped.
    ///
    /// Shared with the workers so they can cull the circuits they generate.
    max_cost: Arc<RwLock<Option<P>>>,
}

impl<P: CircuitCost> PriorityChannelCommunication<P> {
    /// Signal the priority channel to stop.
    ///
    /// This will in turn signal the workers to stop.
    pub fn close(&self) -> Result<(), SendError<Vec<Work<P>>>> {
        self.push.send(Vec::new())
    }

    /// Send a lot of circuits to the priority channel.
    pub fn send(&self, work: Vec<Work<P>>) -> Result<(), SendError<Vec<Work<P>>>> {
        if work.is_empty() {
            return Ok(());
        }
        self.push.send(work)
    }

    /// Receive a circuit from the priority channel.
    ///
    /// Blocks until a circuit is available.
    pub fn recv(&self) -> Result<Work<P>, RecvError> {
        self.pop.recv()
    }

    /// Get the maximum accepted circuit cost.
    ///
    /// This function requires locking, so its value should be cached where
    /// appropriate.
    pub fn max_cost(&self) -> Option<P> {
        self.max_cost.read().as_deref().ok().cloned().flatten()
    }
}

impl<C, P> HugrPriorityChannel<C, P>
where
    C: Fn(&Circuit) -> P + Send + Sync + 'static,
    P: CircuitCost + Send + Sync + 'static,
{
    /// Initialize the queueing system.
    ///
    /// Start the circuit priority queue in a new thread.
    ///
    /// Get back a [`PriorityChannelCommunication`] for adding and removing circuits to/from the queue,
    /// and a channel receiver to receive logging information.
    pub fn init(
        cost_fn: C,
        queue_capacity: usize,
    ) -> (
        PriorityChannelCommunication<P>,
        Receiver<PriorityChannelLog<P>>,
    ) {
        // Shared maximum cost in the queue.
        let max_cost = Arc::new(RwLock::new(None));
        // Channels for pushing and popping circuits from pqueue
        let (tx_push, rx_push) = crossbeam_channel::unbounded();
        let (tx_pop, rx_pop) = crossbeam_channel::bounded(0);
        // Channel for logging results and statistics to the main thread.
        let (tx_log, rx_log) = crossbeam_channel::unbounded();

        let pq = HugrPriorityChannel::new(
            rx_push,
            tx_pop,
            tx_log,
            max_cost.clone(),
            cost_fn,
            queue_capacity,
        );
        pq.run();
        (
            PriorityChannelCommunication {
                push: tx_push,
                pop: rx_pop,
                max_cost,
            },
            rx_log,
        )
    }

    fn new(
        push: Receiver<Vec<Work<P>>>,
        pop: Sender<Work<P>>,
        log: Sender<PriorityChannelLog<P>>,
        max_cost: Arc<RwLock<Option<P>>>,
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
            // Ensure we log the first progress.
            last_progress_log: Instant::now() - std::time::Duration::from_secs(60),
            pq,
            seen_hashes,
            min_cost,
            circ_cnt,
            max_cost,
            local_max_cost: None,
        }
    }

    /// Run the queuer as a thread.
    fn run(mut self) {
        let builder = thread::Builder::new().name("priority queueing".into());
        let _ = builder
            .name("priority-channel".into())
            .spawn(move || {
                'main: loop {
                    while self.pq.is_empty() {
                        let Ok(new_circs) = self.push.recv() else {
                            // Something went wrong
                            break 'main;
                        };
                        if new_circs.is_empty() {
                            // The main thread signalled us to stop.
                            break 'main;
                        }
                        self.enqueue_circs(new_circs);
                    }
                    select! {
                        recv(self.push) -> result => {
                            let Ok(new_circs) = result else {
                                // Something went wrong
                                break 'main;
                            };
                            if new_circs.is_empty() {
                                // The main thread signalled us to stop.
                                break 'main;
                            }
                            self.enqueue_circs(new_circs);
                        }
                        send(self.pop, self.pq.pop().unwrap()) -> result => {
                            if result.is_err() {
                                // Something went wrong.
                                break 'main;
                            }
                            self.update_max_cost();
                        }
                    }
                }
                // Send a last set of logs before terminating.
                self.log
                    .send(PriorityChannelLog::CircuitCount {
                        processed_count: self.circ_cnt,
                        seen_count: self.seen_hashes.len(),
                        queue_length: self.pq.len(),
                    })
                    .unwrap();
            })
            .unwrap();
    }

    /// Add circuits to queue.
    #[tracing::instrument(target = "badger::metrics", skip(self, circs))]
    fn enqueue_circs(&mut self, circs: Vec<Work<P>>) {
        for Work { cost, hash, circ } in circs {
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

            self.pq.push_unchecked(circ, hash, cost);
        }
        self.update_max_cost();

        // This is the result from processing a circuit. Add it to the count.
        self.circ_cnt += 1;
        if Instant::now() - self.last_progress_log > std::time::Duration::from_millis(100) {
            self.log
                .send(PriorityChannelLog::CircuitCount {
                    processed_count: self.circ_cnt,
                    seen_count: self.seen_hashes.len(),
                    queue_length: self.pq.len(),
                })
                .unwrap();
        }
    }

    /// Update the shared `max_cost` value.
    ///
    /// If the priority queue is full, set the `max_cost` to the maximum cost.
    /// Otherwise, leave it as `None`.
    #[inline]
    fn update_max_cost(&mut self) {
        if !self.pq.is_full() || self.pq.is_empty() {
            return;
        }
        let queue_max = self.pq.max_cost().unwrap().clone();
        let local_max = self.local_max_cost.clone();
        if local_max.is_some() && queue_max < local_max.unwrap() {
            self.local_max_cost = Some(queue_max.clone());
            *self.max_cost.write().unwrap() = Some(queue_max);
        }
    }
}
