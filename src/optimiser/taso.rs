//! TASO circuit optimiser.
//!
//! This module implements the TASO circuit optimiser. It relies on a rewriter
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
mod hugr_pchannel;
mod hugr_pqueue;
pub mod log;
mod qtz_circuit;
mod worker;

use crossbeam_channel::select;
pub use eq_circ_class::{load_eccs_json_file, EqCircClass};
use fxhash::FxHashSet;
pub use log::TasoLogger;

use std::fmt;
use std::num::NonZeroUsize;
use std::time::{Duration, Instant};

use hugr::Hugr;

use crate::circuit::CircuitHash;
use crate::optimiser::taso::hugr_pchannel::{HugrPriorityChannel, PriorityChannelLog};
use crate::optimiser::taso::hugr_pqueue::{Entry, HugrPQ};
use crate::optimiser::taso::worker::TasoWorker;
use crate::rewrite::strategy::RewriteStrategy;
use crate::rewrite::Rewriter;

/// The TASO optimiser.
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
/// This optimiser is single-threaded.
///
/// [Quartz]: https://arxiv.org/abs/2204.09033
/// [TASO]: https://dl.acm.org/doi/10.1145/3341301.3359630
#[derive(Clone, Debug)]
pub struct TasoOptimiser<R, S> {
    rewriter: R,
    strategy: S,
}

impl<R, S> TasoOptimiser<R, S> {
    /// Create a new TASO optimiser.
    pub fn new(rewriter: R, strategy: S) -> Self {
        Self { rewriter, strategy }
    }

    fn cost(&self, circ: &Hugr) -> S::Cost
    where
        S: RewriteStrategy,
    {
        self.strategy.circuit_cost(circ)
    }
}

impl<R, S> TasoOptimiser<R, S>
where
    R: Rewriter + Send + Clone + 'static,
    S: RewriteStrategy + Send + Sync + Clone + 'static,
    S::Cost: fmt::Debug + serde::Serialize,
{
    /// Run the TASO optimiser on a circuit.
    ///
    /// A timeout (in seconds) can be provided.
    pub fn optimise(&self, circ: &Hugr, timeout: Option<u64>, n_threads: NonZeroUsize) -> Hugr
    where
        S::Cost: Send + Sync + Clone,
    {
        self.optimise_with_log(circ, Default::default(), timeout, n_threads)
    }

    /// Run the TASO optimiser on a circuit with logging activated.
    ///
    /// A timeout (in seconds) can be provided.
    pub fn optimise_with_log(
        &self,
        circ: &Hugr,
        log_config: TasoLogger,
        timeout: Option<u64>,
        n_threads: NonZeroUsize,
    ) -> Hugr
    where
        S::Cost: Send + Sync + Clone,
    {
        match n_threads.get() {
            1 => self.taso(circ, log_config, timeout),
            _ => self.taso_multithreaded(circ, log_config, timeout, n_threads),
        }
    }

    #[tracing::instrument(target = "taso::metrics", skip(self, circ, logger))]
    fn taso(&self, circ: &Hugr, mut logger: TasoLogger, timeout: Option<u64>) -> Hugr {
        let start_time = Instant::now();

        let mut best_circ = circ.clone();
        let mut best_circ_cost = self.cost(circ);
        logger.log_best(&best_circ_cost);

        // Hash of seen circuits. Dot not store circuits as this map gets huge
        let mut seen_hashes = FxHashSet::default();
        seen_hashes.insert(circ.circuit_hash());

        // The priority queue of circuits to be processed (this should not get big)
        const PRIORITY_QUEUE_CAPACITY: usize = 10_000;
        let cost_fn = {
            let strategy = self.strategy.clone();
            move |circ: &'_ Hugr| strategy.circuit_cost(circ)
        };
        let mut pq = HugrPQ::with_capacity(cost_fn, PRIORITY_QUEUE_CAPACITY);
        pq.push(circ.clone());

        let mut circ_cnt = 1;
        let mut timeout_flag = false;
        while let Some(Entry { circ, cost, .. }) = pq.pop() {
            if cost < best_circ_cost {
                best_circ = circ.clone();
                best_circ_cost = cost;
                logger.log_best(&best_circ_cost);
            }

            let rewrites = self.rewriter.get_rewrites(&circ);
            for new_circ in self.strategy.apply_rewrites(rewrites, &circ) {
                let new_circ_hash = new_circ.circuit_hash();
                if !seen_hashes.insert(new_circ_hash) {
                    // Ignore this circuit: we've already seen it
                    continue;
                }
                circ_cnt += 1;
                logger.log_progress(circ_cnt, Some(pq.len()), seen_hashes.len());
                let new_circ_cost = self.cost(&new_circ);
                pq.push_unchecked(new_circ, new_circ_hash, new_circ_cost);
            }

            if pq.len() >= PRIORITY_QUEUE_CAPACITY {
                // Haircut to keep the queue size manageable
                pq.truncate(PRIORITY_QUEUE_CAPACITY / 2);
            }

            if let Some(timeout) = timeout {
                if start_time.elapsed().as_secs() > timeout {
                    timeout_flag = true;
                    break;
                }
            }
        }

        logger.log_processing_end(circ_cnt, best_circ_cost, false, timeout_flag);
        best_circ
    }

    /// Run the TASO optimiser on a circuit, using multiple threads.
    ///
    /// This is the multi-threaded version of [`taso`]. See [`TasoOptimiser`] for
    /// more details.
    #[tracing::instrument(target = "taso::metrics", skip(self, circ, logger))]
    fn taso_multithreaded(
        &self,
        circ: &Hugr,
        mut logger: TasoLogger,
        timeout: Option<u64>,
        n_threads: NonZeroUsize,
    ) -> Hugr
    where
        S::Cost: Send + Sync + Clone,
    {
        let n_threads: usize = n_threads.get();
        const PRIORITY_QUEUE_CAPACITY: usize = 10_000;

        // multi-consumer priority channel for queuing circuits to be processed by the workers
        let cost_fn = {
            let strategy = self.strategy.clone();
            move |circ: &'_ Hugr| strategy.circuit_cost(circ)
        };
        let mut pq = HugrPriorityChannel::init(cost_fn, PRIORITY_QUEUE_CAPACITY * n_threads);

        let initial_circ_hash = circ.circuit_hash();
        let mut best_circ = circ.clone();
        let mut best_circ_cost = self.cost(&best_circ);

        // Each worker waits for circuits to scan for rewrites using all the
        // patterns and sends the results back to main.
        let joins: Vec<_> = (0..n_threads)
            .map(|i| {
                TasoWorker::spawn(
                    pq.pop.clone().unwrap(),
                    pq.push.clone().unwrap(),
                    self.rewriter.clone(),
                    self.strategy.clone(),
                    Some(format!("taso-worker-{i}")),
                )
            })
            .collect();

        // Queue the initial circuit
        pq.push
            .as_ref()
            .unwrap()
            .send(vec![(initial_circ_hash, circ.clone())])
            .unwrap();
        // Drop our copy of the priority queue channels, so we don't count as a
        // connected worker.
        pq.drop_pop_push();

        // TODO: Report dropped jobs in the queue, so we can check for termination.

        // Deadline for the optimisation timeout
        let timeout_event = match timeout {
            None => crossbeam_channel::never(),
            Some(t) => crossbeam_channel::at(Instant::now() + Duration::from_secs(t)),
        };

        // Main loop: log best circuits as they come in from the priority queue,
        // until the timeout is reached.
        let mut timeout_flag = false;
        loop {
            select! {
                recv(pq.log) -> msg => {
                    match msg {
                        Ok(PriorityChannelLog::NewBestCircuit(circ, cost)) => {
                            best_circ = circ;
                            best_circ_cost = cost;
                            logger.log_best(&best_circ_cost);
                        },
                        Ok(PriorityChannelLog::CircuitCount(circuit_cnt, seen_cnt)) => {
                            logger.log_progress(circuit_cnt, None, seen_cnt);
                        }
                        Err(crossbeam_channel::RecvError) => {
                            eprintln!("Priority queue panicked. Stopping optimisation.");
                            break;
                        }
                    }
                }
                recv(timeout_event) -> _ => {
                    timeout_flag = true;
                    pq.timeout();
                    break;
                }
            }
        }

        // Empty the log from the priority queue and store final circuit count.
        let mut circuit_cnt = None;
        while let Ok(log) = pq.log.recv() {
            match log {
                PriorityChannelLog::NewBestCircuit(circ, cost) => {
                    best_circ = circ;
                    best_circ_cost = cost;
                    logger.log_best(&best_circ_cost);
                }
                PriorityChannelLog::CircuitCount(circ_cnt, _) => {
                    circuit_cnt = Some(circ_cnt);
                }
            }
        }
        logger.log_processing_end(circuit_cnt.unwrap_or(0), best_circ_cost, true, timeout_flag);

        joins.into_iter().for_each(|j| j.join().unwrap());

        best_circ
    }
}

#[cfg(feature = "portmatching")]
mod taso_default {
    use std::io;
    use std::path::Path;

    use hugr::ops::OpType;

    use crate::rewrite::ecc_rewriter::RewriterSerialisationError;
    use crate::rewrite::strategy::NonIncreasingGateCountStrategy;
    use crate::rewrite::ECCRewriter;

    use super::*;

    /// The default TASO optimiser using ECC sets.
    pub type DefaultTasoOptimiser = TasoOptimiser<
        ECCRewriter,
        NonIncreasingGateCountStrategy<fn(&OpType) -> usize, fn(&OpType) -> usize>,
    >;

    impl DefaultTasoOptimiser {
        /// A sane default optimiser using the given ECC sets.
        pub fn default_with_eccs_json_file(eccs_path: impl AsRef<Path>) -> io::Result<Self> {
            let rewriter = ECCRewriter::try_from_eccs_json_file(eccs_path)?;
            let strategy = NonIncreasingGateCountStrategy::default_cx();
            Ok(TasoOptimiser::new(rewriter, strategy))
        }

        /// A sane default optimiser using a precompiled binary rewriter.
        pub fn default_with_rewriter_binary(
            rewriter_path: impl AsRef<Path>,
        ) -> Result<Self, RewriterSerialisationError> {
            let rewriter = ECCRewriter::load_binary(rewriter_path)?;
            let strategy = NonIncreasingGateCountStrategy::default_cx();
            Ok(TasoOptimiser::new(rewriter, strategy))
        }
    }
}
#[cfg(feature = "portmatching")]
pub use taso_default::DefaultTasoOptimiser;

#[cfg(test)]
#[cfg(feature = "portmatching")]
mod tests {
    use hugr::{
        builder::{DFGBuilder, Dataflow, DataflowHugr},
        extension::prelude::QB_T,
        std_extensions::arithmetic::float_types::FLOAT64_TYPE,
        types::FunctionType,
        Hugr,
    };
    use rstest::{fixture, rstest};

    use crate::{extension::REGISTRY, Circuit, T2Op};

    use super::{DefaultTasoOptimiser, TasoOptimiser};

    #[fixture]
    fn rz_rz() -> Hugr {
        let input_t = vec![QB_T, FLOAT64_TYPE, FLOAT64_TYPE];
        let output_t = vec![QB_T];
        let mut h = DFGBuilder::new(FunctionType::new(input_t, output_t)).unwrap();

        let mut inps = h.input_wires();
        let qb = inps.next().unwrap();
        let f1 = inps.next().unwrap();
        let f2 = inps.next().unwrap();

        let res = h.add_dataflow_op(T2Op::RzF64, [qb, f1]).unwrap();
        let qb = res.outputs().next().unwrap();
        let res = h.add_dataflow_op(T2Op::RzF64, [qb, f2]).unwrap();
        let qb = res.outputs().next().unwrap();

        h.finish_hugr_with_outputs([qb], &REGISTRY).unwrap()
    }

    #[fixture]
    fn taso_opt() -> DefaultTasoOptimiser {
        TasoOptimiser::default_with_eccs_json_file("test_files/small_eccs.json").unwrap()
    }

    #[rstest]
    fn rz_rz_cancellation(rz_rz: Hugr, taso_opt: DefaultTasoOptimiser) {
        let opt_rz = taso_opt.optimise(&rz_rz, None, 1.try_into().unwrap());
        let cmds = opt_rz
            .commands()
            .map(|cmd| {
                (
                    cmd.optype().try_into().unwrap(),
                    cmd.inputs().count(),
                    cmd.outputs().count(),
                )
            })
            .collect::<Vec<(T2Op, _, _)>>();
        let exp_cmds = vec![(T2Op::AngleAdd, 2, 1), (T2Op::RzF64, 2, 1)];
        assert_eq!(cmds, exp_cmds);
    }
}
