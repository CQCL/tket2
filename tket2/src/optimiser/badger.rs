//! Badger circuit optimiser.
//!
//! This module implements the Badger circuit optimiser. It relies on a rewriter
//! and a RewriteStrategy instance to repeatedly rewrite a circuit and optimising
//! it according to some cost metric (typically gate count).
//!
//! The optimiser is implemented as a priority queue of circuits to be processed.
//! On top of the queue are the circuits with the lowest cost. They are popped
//! from the queue and replaced by the new circuits obtained from the rewriter
//! and the rewrite strategy. A hash of every circuit computed is stored to
//! detect and ignore duplicates. The priority queue is truncated whenever
//! it gets too large.

mod eq_circ_class;
// mod hugr_pchannel;
mod hugr_pqueue;
pub mod log;
mod qtz_circuit;
// mod worker;

pub use eq_circ_class::{load_eccs_json_file, EqCircClass};
use fxhash::FxHashSet;
pub use log::BadgerLogger;
use portdiff::{GraphView, PortDiff};

use std::collections::BTreeSet;
use std::num::NonZeroUsize;
use std::ops::AddAssign;
use std::time::Instant;

use crate::circuit::{CircuitHash, ToTk2OpIter};
// use crate::optimiser::badger::hugr_pchannel::{HugrPriorityChannel, PriorityChannelLog};
use crate::optimiser::badger::hugr_pqueue::{Entry, HugrPQ};
// use crate::optimiser::badger::worker::BadgerWorker;
use crate::portdiff::{DiffCircuit, DiffRewrite};
use crate::rewrite::strategy::StrategyCost;
use crate::rewrite::Rewriter;
use crate::static_circ::{StaticSizeCircuit, UpdatableHash};

/// Configuration options for the Badger optimiser.
#[derive(Copy, Clone, Debug)]
pub struct BadgerOptions {
    /// The maximum time (in seconds) to run the optimiser.
    ///
    /// Defaults to `None`, which means no timeout.
    pub timeout: Option<u64>,
    /// The maximum time (in seconds) to search for new improvements to the
    /// circuit. If no progress is made in this time, the optimiser will stop.
    ///
    /// Defaults to `None`, which means no timeout.
    pub progress_timeout: Option<u64>,
    /// The maximum number of circuits to process before stopping the optimisation.
    ///
    /// For data parallel multi-threading, (split_circuit=true), applies on a
    /// per-thread basis, otherwise applies globally.
    ///
    /// Defaults to `None`, which means no limit.
    pub max_circuit_count: Option<usize>,
    /// The number of threads to use.
    ///
    /// Defaults to `1`.
    pub n_threads: NonZeroUsize,
    /// Whether to split the circuit into chunks and process each in a separate thread.
    ///
    /// If this option is set to `true`, the optimiser will split the circuit into `n_threads`
    /// chunks.
    ///
    /// If this option is set to `false`, the optimiser will run parallel searches on the whole
    /// circuit.
    ///
    /// Defaults to `false`.
    pub split_circuit: bool,
    /// The maximum size of the circuit candidates priority queue.
    ///
    /// Defaults to `20`.
    pub queue_size: usize,
}

impl Default for BadgerOptions {
    fn default() -> Self {
        Self {
            timeout: Default::default(),
            progress_timeout: Default::default(),
            n_threads: NonZeroUsize::new(1).unwrap(),
            split_circuit: Default::default(),
            queue_size: 20,
            max_circuit_count: None,
        }
    }
}

/// The Badger optimiser.
///
/// Adapted from [Quartz][], and originally [TASO][].
///
/// Using a rewriter and a rewrite strategy, the optimiser
/// will repeatedly rewrite the circuit, optimising the circuit according to
/// the cost function provided.
///
/// Optimisation is done by maintaining a priority queue of circuits and
/// always processing the circuit with the lowest cost first. Rewrites are
/// computed for that circuit and all new circuit obtained are added to the queue.
///
/// There are a single-threaded and two multi-threaded versions of the optimiser,
/// controlled by setting the [`BadgerOptions::n_threads`] and
/// [`BadgerOptions::split_circuit`] fields.
///
/// [Quartz]: https://arxiv.org/abs/2204.09033
/// [TASO]: https://dl.acm.org/doi/10.1145/3341301.3359630
#[derive(Clone, Debug)]
pub struct BadgerOptimiser<R, Cost> {
    rewriter: R,
    cost: Cost,
}

impl<R, Cost> BadgerOptimiser<R, Cost> {
    /// Create a new Badger optimiser.
    pub fn new(rewriter: R, cost: Cost) -> Self {
        Self { rewriter, cost }
    }

    fn cost<C: ToTk2OpIter>(&self, circ: &C) -> Cost::OpCost
    where
        Cost: StrategyCost,
    {
        self.cost.circuit_cost(circ)
    }
}

impl<R, Cost: StrategyCost + Clone> BadgerOptimiser<R, Cost> {
    /// Run the Badger optimiser on a circuit.
    ///
    /// A timeout (in seconds) can be provided.
    pub fn optimise(&self, circ: &StaticSizeCircuit, options: BadgerOptions) -> StaticSizeCircuit
    where
        R: Rewriter<DiffCircuit, CircuitRewrite = DiffRewrite> + Clone,
        // R: Rewriter<StaticSizeCircuit, CircuitRewrite = StaticRewrite<F>> + Clone,
        // F: Fn(StaticQubitIndex) -> StaticQubitIndex,
        Cost::OpCost: serde::Serialize,
    {
        self.optimise_with_log(circ, Default::default(), options)
    }

    /// Run the Badger optimiser on a circuit with logging activated.
    ///
    /// A timeout (in seconds) can be provided.
    pub fn optimise_with_log(
        &self,
        circ: &StaticSizeCircuit,
        log_config: BadgerLogger,
        options: BadgerOptions,
    ) -> StaticSizeCircuit
    where
        R: Rewriter<DiffCircuit, CircuitRewrite = DiffRewrite> + Clone,
        // R: Rewriter<StaticSizeCircuit, CircuitRewrite = StaticRewrite<F>> + Clone,
        // F: Fn(StaticQubitIndex) -> StaticQubitIndex,
        Cost::OpCost: serde::Serialize,
    {
        match options.n_threads.get() {
            1 => {
                let diffs = self.badger_diff(circ, log_config, options);
                // Serialize the diff as JSON and print to stdout
                let json = serde_json::to_string(&diffs).unwrap();
                std::fs::write("diffs.json", json).expect("Unable to write file");
                PortDiff::extract_graph(diffs.sinks().collect()).unwrap()
            }
            _ => {
                // if options.split_circuit {
                //     self.badger_split_multithreaded(circ, log_config, options)
                //         .unwrap()
                // } else {
                //     self.badger_multithreaded(circ, log_config, options)
                // }
                unimplemented!("not implemented multi-threaded version")
            }
        }
    }

    /// Run the Badger optimiser on a circuit, using a single thread.
    // #[tracing::instrument(target = "badger::metrics", skip(self, circ, logger))]
    // fn badger<F>(
    //     &self,
    //     circ: &StaticSizeCircuit,
    //     mut logger: BadgerLogger,
    //     opt: BadgerOptions,
    // ) -> StaticSizeCircuit
    // where
    //     R: Rewriter<StaticSizeCircuit, CircuitRewrite = StaticRewrite<F>> + Clone,
    //     F: Fn(StaticQubitIndex) -> StaticQubitIndex,
    //     Cost::OpCost: serde::Serialize,
    // {
    //     let start_time = Instant::now();
    //     let mut last_best_time = Instant::now();

    //     let circ = circ.to_owned();
    //     let mut best_circ = circ.clone();
    //     let mut best_circ_cost = self.cost(&circ);
    //     // let num_rewrites = best_circ.rewrite_trace().map(|rs| rs.len());
    //     logger.log_best(&best_circ_cost, None);

    //     // Hash of seen circuits. Dot not store circuits as this map gets huge
    //     let hash = circ.circuit_hash().unwrap();
    //     let mut seen_hashes = FxHashSet::default();
    //     seen_hashes.insert(hash);

    //     // The priority queue of circuits to be processed (this should not get big)
    //     let cost_fn = {
    //         let strategy = self.cost.clone();
    //         move |circ: &'_ StaticSizeCircuit| strategy.circuit_cost(circ)
    //     };
    //     let cost = (cost_fn)(&circ);

    //     let mut pq = HugrPQ::new(cost_fn, opt.queue_size);
    //     pq.push_unchecked(circ.to_owned(), hash, cost);

    //     let mut circ_cnt = 0;
    //     let mut timeout_flag = false;
    //     while let Some(Entry { circ, cost, .. }) = pq.pop() {
    //         if cost < best_circ_cost {
    //             best_circ = circ.clone();
    //             best_circ_cost = cost.clone();
    //             // let num_rewrites = best_circ.rewrite_trace().map(|rs| rs.len());
    //             logger.log_best(&best_circ_cost, None);
    //             last_best_time = Instant::now();
    //         }
    //         circ_cnt += 1;

    //         let rewrites = self.rewriter.get_rewrites(&circ);
    //         logger.register_branching_factor(rewrites.len());

    //         // Get combinations of rewrites that can be applied to the circuit,
    //         // and filter them to keep only the ones that
    //         //
    //         // - Don't have a worse cost than the last candidate in the priority queue.
    //         // - Do not invalidate the circuit by creating a loop.
    //         // - We haven't seen yet.
    //         let fast_hasher = UpdatableHash::with_static(&circ);
    //         for rw in rewrites {
    //             let Ok(rw_cost) = self.rewriter.rewrite_cost_delta(&rw, &circ, &self.cost) else {
    //                 continue; // could not compute cost, probably not convex
    //             };
    //             let new_circ_cost = cost.add_delta(&rw_cost);
    //             if !pq.check_accepted(&new_circ_cost) {
    //                 continue;
    //             }

    //             let Ok(new_circ_hash) = fast_hasher.hash_rewrite(&rw) else {
    //                 // The composed rewrites produced a loop.
    //                 //
    //                 // See [https://github.com/CQCL/tket2/discussions/242]
    //                 continue;
    //             };

    //             if !seen_hashes.insert(new_circ_hash) {
    //                 // Ignore this circuit: we've already seen it
    //                 continue;
    //             }

    //             if let Ok(new_circ) = self.rewriter.apply_rewrite(rw, &circ) {
    //                 pq.push_unchecked(new_circ, new_circ_hash, new_circ_cost);
    //                 logger.log_progress(circ_cnt, Some(pq.len()), seen_hashes.len());
    //             }
    //         }

    //         if let Some(timeout) = opt.timeout {
    //             if start_time.elapsed().as_secs() > timeout {
    //                 timeout_flag = true;
    //                 break;
    //             }
    //         }
    //         if let Some(p_timeout) = opt.progress_timeout {
    //             if last_best_time.elapsed().as_secs() > p_timeout {
    //                 timeout_flag = true;
    //                 break;
    //             }
    //         }
    //         if let Some(max_circuit_count) = opt.max_circuit_count {
    //             if seen_hashes.len() >= max_circuit_count {
    //                 timeout_flag = true;
    //                 break;
    //             }
    //         }
    //     }

    //     logger.log_processing_end(
    //         circ_cnt,
    //         Some(seen_hashes.len()),
    //         best_circ_cost,
    //         false,
    //         timeout_flag,
    //         start_time.elapsed(),
    //     );
    //     best_circ
    // }

    fn badger_diff(
        &self,
        circ: &StaticSizeCircuit,
        mut logger: BadgerLogger,
        opt: BadgerOptions,
    ) -> GraphView<StaticSizeCircuit>
    where
        R: Rewriter<DiffCircuit, CircuitRewrite = DiffRewrite> + Clone,
        Cost::OpCost: serde::Serialize,
    {
        let start_time = Instant::now();
        let mut last_best_time = Instant::now();

        let mut circ = circ.clone();
        circ.add_input(); // Add an input and output nodes so that the qubit order is fixed
        let hash = circ.circuit_hash().unwrap();

        let circ = DiffCircuit::from_graph(circ.clone());
        // The list of all "salient" circuits, i.e. those that result from
        // salient rewrites (the best type of rewrite).
        let mut salient_diffs = BTreeSet::from([circ.clone()]);

        // Hash of seen circuits. Dot not store circuits as this map gets huge
        let mut seen_hashes = FxHashSet::default();
        seen_hashes.insert(hash);

        let mut pq = HugrPQ::new(opt.queue_size, |circ: &'_ PortDiff<_>| {
            PortDiff::as_ptr(&circ) as u64
        });
        pq.push(circ.to_owned(), PQCost::default());

        let mut circ_cnt = 0;
        let mut timeout_flag = false;
        while let Some(Entry { circ, cost }) = pq.pop() {
            circ_cnt += 1;

            let rewrites = self.rewriter.get_rewrites(&circ);
            logger.register_branching_factor(rewrites.len());

            // Get combinations of rewrites that can be applied to the circuit,
            // and filter them to keep only the ones that
            //
            // - Don't have a worse cost than the last candidate in the priority queue.
            // - Do not invalidate the circuit by creating a loop.
            // - We haven't seen yet.
            let Ok(extracted) = PortDiff::extract_graph(vec![circ.clone()]) else {
                continue;
            };
            let fast_hasher = UpdatableHash::with_static(&extracted);
            for rw in rewrites {
                let Ok(rw_cost) = self.rewriter.rewrite_cost_delta(&rw, &circ, &self.cost) else {
                    println!("rewrite cost delta failed");
                    continue; // could not compute cost, probably not convex
                };
                let mut new_cost = cost.add_rewrite_cost(rw_cost);
                if !pq.check_accepted(&new_cost) {
                    continue;
                }

                // let Ok(new_circ_hash) = fast_hasher.hash_rewrite(&rw) else {
                //     // The composed rewrites produced a loop.
                //     //
                //     // See [https://github.com/CQCL/tket2/discussions/242]
                //     continue;
                // };

                // if !seen_hashes.insert(new_circ_hash) {
                //     // Ignore this circuit: we've already seen it
                //     continue;
                // }

                if let Ok(new_circ) = self.rewriter.apply_rewrite(rw.clone(), &circ) {
                    // TODO: use updateable hash
                    let Ok(extracted) = PortDiff::extract_graph(vec![new_circ.clone()]) else {
                        continue;
                    };
                    if !extracted.is_acyclic() {
                        continue;
                    }
                    let new_circ_hash = extracted.circuit_hash().unwrap();
                    if !seen_hashes.insert(new_circ_hash) {
                        continue;
                    }
                    if self.cost.is_salient(&new_cost.total_cost_delta) {
                        // The compound rewrite is salient, success! Reset the cost
                        insert_salient(new_circ.clone(), &mut salient_diffs);
                        last_best_time = Instant::now();
                        new_cost = PQCost::zero(); // Reset aggregate cost
                    } else if self.cost.is_salient(&rw_cost) {
                        // The latest rewrite is salient, so maybe this is the
                        // right direction
                        new_cost.n_rewrites_since_salient = 0;
                    }
                    pq.push(new_circ, new_cost);
                    logger.log_progress(circ_cnt, Some(pq.len()), seen_hashes.len());
                }
                println!("pq len: {}", pq.len());
                println!("salient diffs len: {}", salient_diffs.len());
            }

            if let Some(timeout) = opt.timeout {
                if start_time.elapsed().as_secs() > timeout {
                    timeout_flag = true;
                    break;
                }
            }
            if let Some(p_timeout) = opt.progress_timeout {
                if last_best_time.elapsed().as_secs() > p_timeout {
                    timeout_flag = true;
                    break;
                }
            }
            if let Some(max_circuit_count) = opt.max_circuit_count {
                if seen_hashes.len() >= max_circuit_count {
                    timeout_flag = true;
                    break;
                }
            }
        }

        if timeout_flag {
            println!("Timed out");
        }

        logger.log_processing_end(
            circ_cnt,
            Some(seen_hashes.len()),
            (),
            false,
            timeout_flag,
            start_time.elapsed(),
        );
        GraphView::from_sinks(salient_diffs)
    }

    // /// Run the Badger optimiser on a circuit, using multiple threads.
    // ///
    // /// This is the multi-threaded version of [`Self::badger`], using a single
    // /// priority queue and multiple workers to process the circuits in parallel.
    // #[tracing::instrument(target = "badger::metrics", skip(self, circ, logger))]
    // fn badger_multithreaded(
    //     &self,
    //     circ: &Circuit,
    //     mut logger: BadgerLogger,
    //     opt: BadgerOptions,
    // ) -> Circuit
    // where
    //     R: Rewriter<Circuit, Cost> + Send + Clone + Sync + 'static,
    // {
    //     let start_time = Instant::now();
    //     let n_threads: usize = opt.n_threads.get();
    //     let circ = circ.to_owned();

    //     // multi-consumer priority channel for queuing circuits to be processed by the workers
    //     let cost_fn = {
    //         let strategy = self.cost.clone();
    //         move |circ: &'_ Circuit| strategy.circuit_cost(circ)
    //     };
    //     let (pq, rx_log) = HugrPriorityChannel::init(cost_fn.clone(), opt.queue_size);

    //     let initial_circ_hash = circ.circuit_hash().unwrap();
    //     let mut best_circ = circ.clone();
    //     let mut best_circ_cost = self.cost(&best_circ);

    //     // Initialise the work channels and send the initial circuit.
    //     pq.send(vec![Work {
    //         cost: best_circ_cost.clone(),
    //         hash: initial_circ_hash,
    //         circ,
    //     }])
    //     .unwrap();

    //     // Each worker waits for circuits to scan for rewrites using all the
    //     // patterns and sends the results back to main.
    //     let joins: Vec<_> = (0..n_threads)
    //         .map(|i| BadgerWorker::spawn(i, pq.clone(), self.rewriter.clone(), self.cost.clone()))
    //         .collect();

    //     // Deadline for the optimisation timeout
    //     let timeout_event = match opt.timeout {
    //         None => crossbeam_channel::never(),
    //         Some(t) => crossbeam_channel::at(Instant::now() + Duration::from_secs(t)),
    //     };

    //     // Deadline for the timeout when no progress is made
    //     let mut progress_timeout_event = match opt.progress_timeout {
    //         None => crossbeam_channel::never(),
    //         Some(t) => crossbeam_channel::at(Instant::now() + Duration::from_secs(t)),
    //     };

    //     // Main loop: log best circuits as they come in from the priority queue,
    //     // until the timeout is reached.
    //     let mut timeout_flag = false;
    //     let mut processed_count = 0;
    //     let mut seen_count = 0;
    //     loop {
    //         select! {
    //             recv(rx_log) -> msg => {
    //                 match msg {
    //                     Ok(PriorityChannelLog::NewBestCircuit(circ, cost)) => {
    //                         if cost < best_circ_cost {
    //                             best_circ = circ;
    //                             best_circ_cost = cost;
    //                             let num_rewrites = best_circ.rewrite_trace().map(|rs| rs.len());
    //                             logger.log_best(&best_circ_cost, num_rewrites);
    //                             if let Some(t) = opt.progress_timeout {
    //                                 progress_timeout_event = crossbeam_channel::at(Instant::now() + Duration::from_secs(t));
    //                             }
    //                         }
    //                     },
    //                     Ok(PriorityChannelLog::CircuitCount{processed_count: proc, seen_count: seen, queue_length}) => {
    //                         processed_count = proc;
    //                         seen_count = seen;
    //                         if let Some(max_circuit_count) = opt.max_circuit_count {
    //                             if seen_count > max_circuit_count {
    //                                 timeout_flag = true;
    //                                 // Signal the workers to stop.
    //                                 let _ = pq.close();
    //                                 break;
    //                             }
    //                         }
    //                         logger.log_progress(processed_count, Some(queue_length), seen_count);
    //                     }
    //                     Err(crossbeam_channel::RecvError) => {
    //                         logger.log("The priority channel panicked. Stopping Badger optimisation.");
    //                         let _ = pq.close();
    //                         break;
    //                     }
    //                 }
    //             }
    //             recv(timeout_event) -> _ => {
    //                 timeout_flag = true;
    //                 // Signal the workers to stop.
    //                 let _ = pq.close();
    //                 break;
    //             }
    //             recv(progress_timeout_event) -> _ => {
    //                 timeout_flag = true;
    //                 // Signal the workers to stop.
    //                 let _ = pq.close();
    //                 break;
    //             }
    //         }
    //     }

    //     // Empty the log from the priority queue and store final circuit count.
    //     while let Ok(log) = rx_log.recv() {
    //         match log {
    //             PriorityChannelLog::NewBestCircuit(circ, cost) => {
    //                 if cost < best_circ_cost {
    //                     best_circ = circ;
    //                     best_circ_cost = cost;
    //                     let num_rewrites = best_circ.rewrite_trace().map(|rs| rs.len());
    //                     logger.log_best(&best_circ_cost, num_rewrites);
    //                 }
    //             }
    //             PriorityChannelLog::CircuitCount {
    //                 processed_count: proc,
    //                 seen_count: seen,
    //                 queue_length,
    //             } => {
    //                 processed_count = proc;
    //                 seen_count = seen;
    //                 logger.log_progress(processed_count, Some(queue_length), seen_count);
    //             }
    //         }
    //     }
    //     logger.log_processing_end(
    //         processed_count,
    //         Some(seen_count),
    //         best_circ_cost,
    //         true,
    //         timeout_flag,
    //         start_time.elapsed(),
    //     );

    //     joins.into_iter().for_each(|j| j.join().unwrap());

    //     best_circ
    // }

    // /// Run the Badger optimiser on a circuit, with data parallel multithreading.
    // ///
    // /// Split the circuit into chunks and process each in a separate thread.
    // #[tracing::instrument(target = "badger::metrics", skip(self, circ, logger))]
    // fn badger_split_multithreaded(
    //     &self,
    //     circ: &Circuit,
    //     mut logger: BadgerLogger,
    //     opt: BadgerOptions,
    // ) -> Result<Circuit, HugrError>
    // where
    //     R: Rewriter<Circuit> + Send + Clone + Sync + 'static,
    //     S: RewriteStrategy<Circuit, R::CircuitRewrite> + Send + Sync + Clone + 'static,
    //     S::Cost: serde::Serialize + Send + Sync,
    // {
    //     let start_time = Instant::now();
    //     let circ = circ.to_owned();
    //     let circ_cost = self.cost(&circ);
    //     let max_chunk_cost = circ_cost.clone().div_cost(opt.n_threads);
    //     logger.log(format!(
    //         "Splitting circuit with cost {:?} into chunks of at most {max_chunk_cost:?}.",
    //         circ_cost.clone()
    //     ));
    //     let mut chunks =
    //         CircuitChunks::split_with_cost(&circ, max_chunk_cost, |op| self.strategy.op_cost(op));

    //     let num_rewrites = circ.rewrite_trace().map(|rs| rs.len());
    //     logger.log_best(circ_cost.clone(), num_rewrites);

    //     let (joins, rx_work): (Vec<_>, Vec<_>) = chunks
    //         .par_iter_mut()
    //         .enumerate()
    //         .map(|(i, chunk)| {
    //             let (tx, rx) = crossbeam_channel::unbounded();
    //             let badger = self.clone();
    //             let chunk = mem::take(chunk);
    //             let chunk_cx_cost = chunk.circuit_cost(|op| self.strategy.op_cost(op));
    //             logger.log(format!("Chunk {i} has {chunk_cx_cost:?} CX gates",));
    //             let join = thread::Builder::new()
    //                 .name(format!("chunk-{}", i))
    //                 .spawn(move || {
    //                     let res = badger.optimise(
    //                         &chunk,
    //                         BadgerOptions {
    //                             n_threads: NonZeroUsize::new(1).unwrap(),
    //                             split_circuit: false,
    //                             ..opt
    //                         },
    //                     );
    //                     tx.send(res).unwrap();
    //                 })
    //                 .unwrap();
    //             (join, rx)
    //         })
    //         .unzip();

    //     for i in 0..chunks.len() {
    //         let res = rx_work[i]
    //             .recv()
    //             .unwrap_or_else(|_| panic!("Worker thread panicked"));
    //         chunks[i] = res;
    //     }

    //     let best_circ = chunks.reassemble()?;
    //     let best_circ_cost = self.cost(&best_circ);
    //     if best_circ_cost.clone() < circ_cost {
    //         let num_rewrites = best_circ.rewrite_trace().map(|rs| rs.len());
    //         logger.log_best(best_circ_cost.clone(), num_rewrites);
    //     }

    //     logger.log_processing_end(
    //         opt.n_threads.get(),
    //         None,
    //         best_circ_cost,
    //         true,
    //         false,
    //         start_time.elapsed(),
    //     );
    //     joins.into_iter().for_each(|j| j.join().unwrap());

    //     Ok(best_circ)
    // }
}

fn insert_salient(
    new_circ: PortDiff<StaticSizeCircuit>,
    salient_diffs: &mut BTreeSet<PortDiff<StaticSizeCircuit>>,
) {
    // TODO: squash the diffs into a single one
    // Find the set of salient diffs that are the direct predecessors of new_circ
    salient_diffs.insert(new_circ);
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
struct PQCost<C> {
    total_cost_delta: C,
    n_rewrites_since_salient: usize,
}

impl<C: Ord + Copy> PartialOrd for PQCost<C> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl<C: Ord + Copy> Ord for PQCost<C> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        let key = |cost: &Self| (cost.n_rewrites_since_salient, cost.total_cost_delta);
        key(self).cmp(&key(other))
    }
}

impl<C> PQCost<C> {
    fn zero() -> Self
    where
        C: Default,
    {
        Self::default()
    }

    fn add_rewrite_cost(&self, rewrite_cost: C) -> Self
    where
        C: AddAssign + Copy,
    {
        let mut total_cost_delta = self.total_cost_delta;
        total_cost_delta += rewrite_cost;
        Self {
            n_rewrites_since_salient: self.n_rewrites_since_salient + 1,
            total_cost_delta,
        }
    }
}

#[cfg(feature = "portmatching")]
mod badger_default {
    use std::io;
    use std::path::Path;

    use hugr::ops::OpType;

    use crate::portdiff::DiffCircuitMatcher;
    use crate::portmatching::CircuitMatcher;
    use crate::rewrite::ecc_rewriter::RewriterSerialisationError;
    use crate::rewrite::strategy::LexicographicCostFunction;
    use crate::rewrite::ECCRewriter;
    use crate::static_circ::StaticSizeCircuit;
    use crate::Tk2Op;

    use super::*;

    pub type StrategyCost = LexicographicCostFunction<fn(Tk2Op) -> usize, 2>;

    /// The default Badger optimiser using ECC sets.
    pub type DefaultBadgerOptimiser =
        BadgerOptimiser<ECCRewriter<CircuitMatcher, StaticSizeCircuit>, StrategyCost>;

    /// The portdiff Badger optimiser using ECC sets.
    pub type DiffBadgerOptimiser =
        BadgerOptimiser<ECCRewriter<DiffCircuitMatcher, StaticSizeCircuit>, StrategyCost>;

    impl DefaultBadgerOptimiser {
        /// A sane default optimiser using the given ECC sets.
        pub fn default_with_eccs_json_file(eccs_path: impl AsRef<Path>) -> io::Result<Self> {
            let rewriter = ECCRewriter::<CircuitMatcher, _>::try_from_eccs_json_file(eccs_path)?;
            let strategy = LexicographicCostFunction::default_cx();
            Ok(BadgerOptimiser::new(rewriter, strategy))
        }

        /// A sane default optimiser using a precompiled binary rewriter.
        #[cfg(feature = "binary-eccs")]
        pub fn default_with_rewriter_binary(
            rewriter_path: impl AsRef<Path>,
        ) -> Result<Self, RewriterSerialisationError> {
            let rewriter = ECCRewriter::load_binary(rewriter_path)?;
            let strategy = LexicographicCostFunction::default_cx();
            Ok(BadgerOptimiser::new(rewriter, strategy))
        }
    }

    impl DiffBadgerOptimiser {
        /// A sane default optimiser using the given ECC sets.
        pub fn diff_with_eccs_json_file(eccs_path: impl AsRef<Path>) -> io::Result<Self> {
            let rewriter =
                ECCRewriter::<DiffCircuitMatcher, _>::try_from_eccs_json_file(eccs_path)?;
            let strategy = LexicographicCostFunction::default_cx();
            Ok(BadgerOptimiser::new(rewriter, strategy))
        }

        /// A sane default optimiser using a precompiled binary rewriter.
        #[cfg(feature = "binary-eccs")]
        pub fn diff_with_rewriter_binary(
            rewriter_path: impl AsRef<Path>,
        ) -> Result<Self, RewriterSerialisationError> {
            let rewriter = ECCRewriter::load_binary(rewriter_path)?;
            let strategy = LexicographicCostFunction::default_cx();
            Ok(BadgerOptimiser::new(rewriter, strategy))
        }
    }
}

#[cfg(feature = "portmatching")]
pub use badger_default::{DefaultBadgerOptimiser, DiffBadgerOptimiser};

// use self::hugr_pchannel::Work;

#[cfg(test)]
#[cfg(feature = "portmatching")]
mod tests {
    use hugr::{
        builder::{DFGBuilder, Dataflow, DataflowHugr},
        extension::prelude::QB_T,
        std_extensions::arithmetic::float_types::FLOAT64_TYPE,
        types::Signature,
    };
    use rstest::{fixture, rstest};

    use crate::serialize::load_tk1_json_str;
    use crate::{extension::REGISTRY, Circuit, Tk2Op};
    use crate::{optimiser::badger::BadgerOptions, static_circ::StaticSizeCircuit};

    use super::{BadgerOptimiser, DefaultBadgerOptimiser};

    /// Simplified description of the circuit's commands.
    fn gates(circ: &Circuit) -> Vec<Tk2Op> {
        circ.commands()
            .map(|cmd| cmd.optype().try_into().unwrap())
            .collect()
    }

    #[fixture]
    fn rz_rz() -> StaticSizeCircuit {
        let input_t = vec![QB_T, FLOAT64_TYPE, FLOAT64_TYPE];
        let output_t = vec![QB_T];
        let mut h = DFGBuilder::new(Signature::new(input_t, output_t)).unwrap();

        let mut inps = h.input_wires();
        let qb = inps.next().unwrap();
        let f1 = inps.next().unwrap();
        let f2 = inps.next().unwrap();

        let res = h.add_dataflow_op(Tk2Op::RzF64, [qb, f1]).unwrap();
        let qb = res.outputs().next().unwrap();
        let res = h.add_dataflow_op(Tk2Op::RzF64, [qb, f2]).unwrap();
        let qb = res.outputs().next().unwrap();

        let circ: Circuit = h.finish_hugr_with_outputs([qb], &REGISTRY).unwrap().into();
        StaticSizeCircuit::try_from(&circ).unwrap()
    }

    /// This hugr corresponds to the qasm circuit:
    ///
    /// ```skip
    /// OPENQASM 2.0;
    /// include "qelib1.inc";
    ///
    /// qreg q[5];
    /// cx q[4],q[1];
    /// cx q[3],q[4];
    /// cx q[1],q[2];
    /// cx q[4],q[0];
    /// u3(0.5*pi,0.0*pi,0.5*pi) q[1];
    /// cx q[0],q[2];
    /// cx q[3],q[1];
    /// cx q[0],q[2];
    /// ```
    const NON_COMPOSABLE: &str = r#"{"phase":"0.0","commands":[{"op":{"type":"CX","n_qb":2,"signature":["Q","Q"]},"args":[["q",[4]],["q",[1]]]},{"op":{"type":"CX","n_qb":2,"signature":["Q","Q"]},"args":[["q",[1]],["q",[2]]]},{"op":{"type":"U3","params":["0.5","0","0.5"],"signature":["Q"]},"args":[["q",[1]]]},{"op":{"type":"CX","n_qb":2,"signature":["Q","Q"]},"args":[["q",[3]],["q",[4]]]},{"op":{"type":"CX","n_qb":2,"signature":["Q","Q"]},"args":[["q",[4]],["q",[0]]]},{"op":{"type":"CX","n_qb":2,"signature":["Q","Q"]},"args":[["q",[0]],["q",[2]]]},{"op":{"type":"CX","n_qb":2,"signature":["Q","Q"]},"args":[["q",[0]],["q",[2]]]},{"op":{"type":"CX","n_qb":2,"signature":["Q","Q"]},"args":[["q",[3]],["q",[1]]]}],"qubits":[["q",[0]],["q",[1]],["q",[2]],["q",[3]],["q",[4]]],"bits":[],"implicit_permutation":[[["q",[0]],["q",[0]]],[["q",[1]],["q",[1]]],[["q",[2]],["q",[2]]],[["q",[3]],["q",[3]]],[["q",[4]],["q",[4]]]]}"#;

    /// A circuit that would trigger non-composable rewrites, if we applied them blindly from nam_6_3 matches.
    #[fixture]
    fn non_composable_rw_hugr() -> Circuit {
        load_tk1_json_str(NON_COMPOSABLE).unwrap()
    }

    /// A badger optimiser using a reduced set of rewrite rules.
    #[fixture]
    fn badger_opt_json() -> DefaultBadgerOptimiser {
        BadgerOptimiser::default_with_eccs_json_file("../test_files/eccs/small_eccs.json").unwrap()
    }

    /// A badger optimiser using a reduced set of rewrite rules.
    #[fixture]
    fn badger_opt_compiled() -> DefaultBadgerOptimiser {
        BadgerOptimiser::default_with_rewriter_binary("../test_files/eccs/small_eccs.rwr").unwrap()
    }

    /// A badger optimiser using the complete nam_6_3 rewrite set.
    ///
    /// NOTE: This takes a few seconds to load.
    /// Use [`badger_opt`] if possible.
    #[fixture]
    fn badger_opt_full() -> DefaultBadgerOptimiser {
        BadgerOptimiser::default_with_rewriter_binary("../test_files/eccs/nam_6_3.rwr").unwrap()
    }

    // #[rstest]
    // #[case::compiled(badger_opt_compiled())]
    // #[case::json(badger_opt_json())]
    // fn rz_rz_cancellation(rz_rz: StaticSizeCircuit, #[case] badger_opt: DefaultBadgerOptimiser) {
    //     let opt_rz = badger_opt.optimise(
    //         &rz_rz,
    //         BadgerOptions {
    //             queue_size: 4,
    //             ..Default::default()
    //         },
    //     );
    //     // Rzs combined into a single one.
    //     assert_eq!(gates(&opt_rz), vec![Tk2Op::AngleAdd, Tk2Op::RzF64]);
    // }

    // #[rstest]
    // #[case::compiled(badger_opt_compiled())]
    // #[case::json(badger_opt_json())]
    // fn rz_rz_cancellation_parallel(rz_rz: Circuit, #[case] badger_opt: DefaultBadgerOptimiser) {
    //     let mut opt_rz = badger_opt.optimise(
    //         &rz_rz,
    //         BadgerOptions {
    //             timeout: Some(0),
    //             n_threads: 2.try_into().unwrap(),
    //             queue_size: 4,
    //             ..Default::default()
    //         },
    //     );
    //     opt_rz.hugr_mut().update_validate(&REGISTRY).unwrap();
    // }

    // #[rstest]
    // #[case::compiled(badger_opt_compiled())]
    // #[case::json(badger_opt_json())]
    // fn rz_rz_cancellation_split_parallel(
    //     rz_rz: Circuit,
    //     #[case] badger_opt: DefaultBadgerOptimiser,
    // ) {
    //     let mut opt_rz = badger_opt.optimise(
    //         &rz_rz,
    //         BadgerOptions {
    //             timeout: Some(0),
    //             n_threads: 2.try_into().unwrap(),
    //             queue_size: 4,
    //             split_circuit: true,
    //             ..Default::default()
    //         },
    //     );
    //     opt_rz.hugr_mut().update_validate(&REGISTRY).unwrap();
    //     assert_eq!(opt_rz.commands().count(), 2);
    // }

    // #[rstest]
    // #[ignore = "Loading the ECC set is really slow (~5 seconds)"]
    // fn non_composable_rewrites(
    //     non_composable_rw_hugr: Circuit,
    //     badger_opt_full: DefaultBadgerOptimiser,
    // ) {
    //     let mut opt = badger_opt_full.optimise(
    //         &non_composable_rw_hugr,
    //         BadgerOptions {
    //             timeout: Some(0),
    //             queue_size: 4,
    //             ..Default::default()
    //         },
    //     );
    //     // No rewrites applied.
    //     opt.hugr_mut().update_validate(&REGISTRY).unwrap();
    // }

    #[test]
    fn load_precompiled_bin() {
        let opt =
            BadgerOptimiser::default_with_rewriter_binary("../test_files/eccs/small_eccs.rwr");
        opt.unwrap();
    }
}
