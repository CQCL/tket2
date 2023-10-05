//! Distributed workers for the taso optimiser.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread::{self, JoinHandle};

use crossbeam_channel::{RecvError, SendError, TryRecvError};
use fxhash::FxHashSet;
use hugr::Hugr;

use crate::circuit::cost::CircuitCost;
use crate::circuit::CircuitHash;
use crate::rewrite::strategy::RewriteStrategy;
use crate::rewrite::Rewriter;

use super::hugr_pqueue::{Entry, HugrPQ};

/// A unit of work for a worker, consisting of a circuit to process and its
/// hash.
pub type Work = (u64, Hugr);

/// A worker that processes circuits for the TASO optimiser.
pub struct TasoWorker<R, S, C, P: Ord> {
    /// The worker ID. Used to distribute work.
    id: usize,
    /// The channel to receive work.
    pop: crossbeam_channel::Receiver<Vec<Work>>,
    /// A group of channels to send work.
    push: WorkChannel,
    /// A channel to log results and processing stats.
    log: crossbeam_channel::Sender<WorkerLog<P>>,
    /// A shared atomic with the current maximum cost in the queue.
    /// Other workers will avoid sending circuits with a cost higher than this.
    max_cost: Arc<AtomicUsize>,
    /// The rewriter to use.
    rewriter: R,
    /// The rewrite strategy to use.
    strategy: S,
    // The queue capacity. Queue size is halved when it exceeds this.
    queue_capacity: usize,
    // The priority queue data structure.
    pq: HugrPQ<P, C>,
    // The set of hashes we've seen.
    seen_hashes: FxHashSet<u64>,
    // The minimum cost we've seen.
    min_cost: Option<P>,
    // The number of circuits we've seen (for logging).
    circ_cnt: usize,
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
        pop: crossbeam_channel::Receiver<Vec<Work>>,
        push: WorkChannel,
        log: crossbeam_channel::Sender<WorkerLog<P>>,
        rewriter: R,
        strategy: S,
        cost_fn: C,
        queue_capacity: usize,
    ) -> JoinHandle<()> {
        let name = format!("TasoWorker-{id}");
        thread::Builder::new()
            .name(name)
            .spawn(move || {
                let max_cost = push.worker_max_costs[id].clone();
                let mut worker = Self {
                    id,
                    pop,
                    push,
                    log,
                    max_cost,
                    rewriter,
                    strategy,
                    queue_capacity,
                    pq: HugrPQ::with_capacity(cost_fn, queue_capacity),
                    seen_hashes: FxHashSet::default(),
                    min_cost: None,
                    circ_cnt: 0,
                };
                worker.run_loop(queue_capacity)
            })
            .unwrap()
    }

    /// Main loop of the worker.
    ///
    /// Processes work until the main thread closes the channel send or receive
    /// channel.
    #[tracing::instrument(target = "taso::metrics", skip(self))]
    fn run_loop(&mut self, queue_capacity: usize) {
        'main: loop {
            // Receive work and add it to the queue.
            // If there is no work in the queue, we block until we receive some.
            while self.pq.is_empty() {
                match self.pop.recv() {
                    Ok(new_circs) => {
                        // The main thread signalled us to stop.
                        if new_circs.is_empty() {
                            break 'main;
                        }
                        self.enqueue_circs(new_circs);
                    }
                    Err(RecvError) => {
                        // Something went wrong.
                        break 'main;
                    }
                }
            }
            'recv: loop {
                match self.pop.try_recv() {
                    Ok(new_circs) => {
                        // The main thread signalled us to stop.
                        if new_circs.is_empty() {
                            break 'main;
                        }
                        self.enqueue_circs(new_circs);
                    }
                    Err(TryRecvError::Empty) => {
                        // No new work available, continue processing the queue.
                        break 'recv;
                    }
                    Err(TryRecvError::Disconnected) => {
                        // Something went wrong.
                        break 'main;
                    }
                }
            }

            //tracing::info!("processing circ. queue size: {}", self.pq.len());

            // Process the next item in the queue.
            let Entry { circ, .. } = self.pq.pop().unwrap();
            let new_circs = self.process_circ(circ);
            let send = tracing::trace_span!(target: "taso::metrics", "TasoWorker::send_result")
                .in_scope(|| self.push.send(new_circs, Some(self.id)));
            match send {
                Ok(local_work) => {
                    self.enqueue_circs(local_work);
                }
                Err(_) => {
                    // Something went wrong.
                    break 'main;
                }
            }
        }
        self.log_counts();
    }

    /// Process the next circuit in the queue, applying the rewrite strategy.
    #[tracing::instrument(target = "taso::metrics", skip_all)]
    fn process_circ(&self, circ: Hugr) -> Vec<Work> {
        let rewrites = self.rewriter.get_rewrites(&circ);
        let circs = self.strategy.apply_rewrites(rewrites, &circ);
        circs.into_iter().map(|c| (c.circuit_hash(), c)).collect()
    }

    /// Add circuits to queue.
    #[tracing::instrument(target = "taso::metrics", skip(self, circs))]
    #[inline]
    fn enqueue_circs(&mut self, circs: Vec<(u64, Hugr)>) {
        let max_cost = self
            .pq
            .max_cost()
            .map(CircuitCost::as_usize)
            .unwrap_or(usize::MAX);

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
                    .send(WorkerLog::NewBestCircuit(circ.clone(), cost.clone()))
                    .unwrap();
            }

            // Skip this circuit if it is too expensive.
            if cost.as_usize() > max_cost {
                continue;
            }

            if self.pq.is_empty() {
                self.max_cost.store(cost.as_usize(), Ordering::Relaxed);
            }

            self.circ_cnt += 1;
            self.pq.push_unchecked(circ, hash, cost);

            if self.circ_cnt % 1000 == 0 {
                self.log_counts();
            }
        }

        // If the queue got too big, truncate it.
        if self.pq.len() >= self.queue_capacity {
            self.pq.truncate(self.queue_capacity / 2);
            let new_max_cost = self.pq.max_cost().unwrap().as_usize();

            // Update the upper bound on the cost of circuits we will accept.
            // Note that some circuits may have been added to the queue with a
            // cost higher than the new max cost. We don't remove them, but
            // they will be ignored.
            if new_max_cost < max_cost {
                self.max_cost.store(new_max_cost, Ordering::Relaxed);
            }
        }
    }

    /// Log the number of circuits we have seen.
    fn log_counts(&self) {
        self.log
            .send(WorkerLog::CircuitCount {
                worker_id: self.id,
                processed_count: self.circ_cnt,
                seen_count: self.seen_hashes.len(),
            })
            .unwrap();
    }
}

/// A channel to send work to workers.
///
/// Distributes the work pseudo-randomly using circuit hashes.
#[derive(Debug, Clone, Default)]
pub struct WorkChannel {
    worker_channels: Vec<crossbeam_channel::Sender<Vec<Work>>>,
    pub(self) worker_max_costs: Vec<Arc<AtomicUsize>>,
}

impl WorkChannel {
    /// Create a new work channel with a given number of workers.
    pub fn new(num_workers: usize) -> (Self, Vec<crossbeam_channel::Receiver<Vec<Work>>>) {
        let mut worker_channels = Vec::with_capacity(num_workers);
        let mut rx_channels = Vec::with_capacity(num_workers);
        let mut worker_max_costs = Vec::with_capacity(num_workers);
        for _ in 0..num_workers {
            let (tx, rx) = crossbeam_channel::unbounded();
            worker_channels.push(tx);
            rx_channels.push(rx);
            worker_max_costs.push(Arc::new(AtomicUsize::new(usize::MAX)));
        }
        (
            Self {
                worker_channels,
                worker_max_costs,
            },
            rx_channels,
        )
    }

    /// Send a lot of circuits to the workers.
    ///
    /// Returns a vector with circuits for the local thread.
    pub fn send(
        &self,
        work: Vec<Work>,
        local_id: Option<usize>,
    ) -> Result<Vec<Work>, SendError<Vec<Work>>> {
        let mut split_work = vec![Vec::new(); self.worker_channels.len()];
        let mut local_work = Vec::new();
        for (hash, circ) in work {
            let worker = hash as usize % self.worker_channels.len();
            split_work[worker].push((hash, circ));
        }
        for (worker, work) in split_work.into_iter().enumerate() {
            if work.is_empty() {
                continue;
            }
            match Some(worker) == local_id {
                true => local_work = work,
                false => self.worker_channels[worker].send(work)?,
            }
        }
        Ok(local_work)
    }

    /// Signal the workers to stop.
    pub fn close(&self) {
        for channel in &self.worker_channels {
            channel.send(Vec::new()).unwrap();
        }
    }
}

/// Logging information from the priority channel.
pub enum WorkerLog<C> {
    NewBestCircuit(Hugr, C),
    CircuitCount {
        worker_id: usize,
        processed_count: usize,
        seen_count: usize,
    },
}
