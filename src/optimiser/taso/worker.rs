//! Distributed workers for the taso optimiser.

use std::sync::{Arc, RwLock};
use std::thread::{self, JoinHandle};
use std::time::Instant;

use crossbeam_channel::{RecvError, SendError, TryRecvError};
use fxhash::FxHashSet;
use hugr::Hugr;

use crate::circuit::cost::CircuitCost;
use crate::circuit::CircuitHash;
use crate::optimiser::taso::log::PROGRESS_TARGET;
use crate::rewrite::strategy::RewriteStrategy;
use crate::rewrite::Rewriter;

use super::hugr_pqueue::{Entry, HugrPQ};

/// A unit of work for a worker, consisting of a circuit to process, along its
/// hash and cost.
pub type Work<P> = (P, u64, Hugr);

/// A worker that processes circuits for the TASO optimiser.
pub struct TasoWorker<R, S, C, P: Ord> {
    /// The worker ID. Used to distribute work.
    id: usize,
    /// The channel to receive work.
    pop: crossbeam_channel::Receiver<Vec<Work<P>>>,
    /// A group of channels to send work.
    push: WorkChannel<P>,
    /// A channel to log results and processing stats.
    log: crossbeam_channel::Sender<WorkerLog<P>>,
    /// A shared atomic with the current maximum cost in the queue.
    /// Other workers will avoid sending circuits with a cost higher than this.
    max_cost: Arc<RwLock<Option<P>>>,
    /// The rewriter to use.
    rewriter: R,
    /// The rewrite strategy to use.
    strategy: S,
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
        pop: crossbeam_channel::Receiver<Vec<Work<P>>>,
        push: WorkChannel<P>,
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
                    pq: HugrPQ::new(cost_fn, queue_capacity),
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
    fn process_circ(&mut self, circ: Hugr) -> Vec<Work<P>> {
        self.circ_cnt += 1;
        let rewrites = self.rewriter.get_rewrites(&circ);
        let circs = self.strategy.apply_rewrites(rewrites, &circ);
        circs
            .into_iter()
            .map(|c| ((self.pq.cost_fn)(&c), c.circuit_hash(), c))
            .collect()
    }

    /// Add circuits to queue.
    #[tracing::instrument(target = "taso::metrics", skip(self, circs))]
    #[inline]
    fn enqueue_circs(&mut self, circs: Vec<(P, u64, Hugr)>) {
        let max_cost = self.pq.max_cost().cloned();

        for (cost, hash, circ) in circs {
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

            self.pq.push_unchecked(circ, hash, cost);
        }
        self.log_counts();

        // Update the upper bound on the cost of circuits we will accept.
        // Note that some circuits may have been added to the queue with a
        // cost higher than the new max cost. We don't remove them, but
        // they will be ignored.
        let Some(new_max_cost) = self.pq.max_cost().cloned() else {
            return;
        };
        if max_cost.is_none() || new_max_cost < max_cost.unwrap() {
            let mut lock = self.max_cost.write().unwrap();
            *lock = Some(new_max_cost);
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
pub struct WorkChannel<P> {
    worker_channels: Vec<crossbeam_channel::Sender<Vec<Work<P>>>>,
    pub(self) worker_max_costs: Vec<Arc<RwLock<Option<P>>>>,
}

impl<P: CircuitCost> WorkChannel<P> {
    /// Create a new work channel with a given number of workers.
    pub fn new(num_workers: usize) -> (Self, Vec<crossbeam_channel::Receiver<Vec<Work<P>>>>) {
        let mut worker_channels = Vec::with_capacity(num_workers);
        let mut rx_channels = Vec::with_capacity(num_workers);
        let mut worker_max_costs = Vec::with_capacity(num_workers);
        for _ in 0..num_workers {
            let (tx, rx) = crossbeam_channel::unbounded();
            worker_channels.push(tx);
            rx_channels.push(rx);
            worker_max_costs.push(Arc::new(RwLock::new(None)));
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
        work: Vec<Work<P>>,
        local_id: Option<usize>,
    ) -> Result<Vec<Work<P>>, SendError<Vec<Work<P>>>> {
        let mut split_work = vec![Vec::new(); self.worker_channels.len()];
        let mut local_work = Vec::new();
        // Cache with the maximum cost for each worker.
        let mut workers_max_cost = vec![None; self.worker_channels.len()];
        for (cost, hash, circ) in work {
            let worker = hash as usize % self.worker_channels.len();
            let max_cost = workers_max_cost[worker]
                .get_or_insert_with(|| {
                    self.worker_max_costs[worker]
                        .read()
                        .as_deref()
                        .cloned()
                        .unwrap_or_default()
                })
                .clone();
            if max_cost.is_none() || cost <= max_cost.unwrap() {
                split_work[worker].push((cost, hash, circ));
            }
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
pub enum WorkerLog<P> {
    NewBestCircuit(Hugr, P),
    CircuitCount {
        worker_id: usize,
        processed_count: usize,
        seen_count: usize,
    },
}
