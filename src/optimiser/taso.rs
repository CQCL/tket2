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
//!
//! There are currently two implementations: a single-threaded one ([`taso`])
//! and a multi-threaded one ([`taso_mpsc`]).

mod eq_circ_class;
mod hugr_pq;
mod qtz_circuit;

pub use eq_circ_class::{load_eccs_json_file, EqCircClass};

use std::sync::mpsc::{self, Receiver, SyncSender};
use std::thread::{self, JoinHandle};
use std::time::Instant;
use std::{fs, io};

use fxhash::FxHashSet;
use hugr::{Hugr, HugrView};
use itertools::{izip, Itertools};

use crate::circuit::{Circuit, CircuitHash};
use crate::json::save_tk1_json_writer;
use crate::rewrite::strategy::{self, ExhaustiveRewriteStrategy, RewriteStrategy};
use crate::rewrite::{ECCRewriter, Rewriter};
use hugr_pq::{Entry, HugrPQ};

/// Logging configuration for the TASO optimiser.
#[derive(Default)]
pub struct LogConfig<'w> {
    final_circ_json: Option<Box<dyn io::Write + 'w>>,
    circ_candidates_csv: Option<Box<dyn io::Write + 'w>>,
    progress_log: Option<Box<dyn io::Write + 'w>>,
}

impl<'w> LogConfig<'w> {
    /// Create a new logging configuration.
    ///
    /// Three writer objects must be provided:
    /// - best_circ_json: for the final optimised circuit, in TK1 JSON format,
    /// - circ_candidates_csv: for a log of the successive best candidate circuits,
    /// - progress_log: for a log of the progress of the optimisation.
    pub fn new(
        best_circ_json: impl io::Write + 'w,
        circ_candidates_csv: impl io::Write + 'w,
        progress_log: impl io::Write + 'w,
    ) -> Self {
        Self {
            final_circ_json: Some(Box::new(best_circ_json)),
            circ_candidates_csv: Some(Box::new(circ_candidates_csv)),
            progress_log: Some(Box::new(progress_log)),
        }
    }
}

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
/// This is the single-threaded version of the optimiser. See
/// [`TasoMpscOptimiser`] for the multi-threaded version.
///
/// [Quartz]: https://arxiv.org/abs/2204.09033
/// [TASO]: https://dl.acm.org/doi/10.1145/3341301.3359630
pub struct TasoOptimiser<R, S, C> {
    rewriter: R,
    strategy: S,
    cost: C,
}

impl<R, S, C> TasoOptimiser<R, S, C> {
    /// Create a new TASO optimiser.
    pub fn new(rewriter: R, strategy: S, cost: C) -> Self {
        Self {
            rewriter,
            strategy,
            cost,
        }
    }

    /// Run the TASO optimiser on a circuit.
    ///
    /// A timeout (in seconds) can be provided.
    pub fn optimise(&self, circ: &Hugr, timeout: Option<u64>) -> Hugr
    where
        R: Rewriter,
        S: RewriteStrategy,
        C: Fn(&Hugr) -> usize,
    {
        taso(
            circ,
            &self.rewriter,
            &self.strategy,
            &self.cost,
            Default::default(),
            timeout,
        )
    }

    /// Run the TASO optimiser on a circuit with logging activated.
    ///
    /// A timeout (in seconds) can be provided.
    pub fn optimise_with_log(
        &self,
        circ: &Hugr,
        log_config: LogConfig,
        timeout: Option<u64>,
    ) -> Hugr
    where
        R: Rewriter,
        S: RewriteStrategy,
        C: Fn(&Hugr) -> usize,
    {
        taso(
            circ,
            &self.rewriter,
            &self.strategy,
            &self.cost,
            log_config,
            timeout,
        )
    }

    /// Run the TASO optimiser on a circuit with default logging.
    ///
    /// The following files will be created:
    ///  - `final_circ.json`: the final optimised circuit, in TK1 JSON format,
    ///  - `best_circs.csv`: a log of the successive best candidate circuits,
    ///  - `taso-optimisation.log`: a log of the progress of the optimisation.
    ///
    /// If the creation of any of these files failes, an error is returned.
    ///
    /// A timeout (in seconds) can be provided.
    pub fn optimise_with_default_log(&self, circ: &Hugr, timeout: Option<u64>) -> io::Result<Hugr>
    where
        R: Rewriter,
        S: RewriteStrategy,
        C: Fn(&Hugr) -> usize,
    {
        let final_circ_json = fs::File::create("final_circ.json")?;
        let circ_candidates_csv = fs::File::create("best_circs.csv")?;
        let progress_log = fs::File::create("taso-optimisation.log")?;
        let log_config = LogConfig::new(final_circ_json, circ_candidates_csv, progress_log);
        Ok(self.optimise_with_log(circ, log_config, timeout))
    }
}

impl TasoOptimiser<ECCRewriter, ExhaustiveRewriteStrategy, fn(&Hugr) -> usize> {
    /// A sane default optimiser using the given ECC sets.
    pub fn default_with_eccs_json_file(eccs_path: impl AsRef<std::path::Path>) -> Self {
        let rewriter = ECCRewriter::from_eccs_json_file(eccs_path);
        let strategy = strategy::ExhaustiveRewriteStrategy::default();
        Self::new(rewriter, strategy, |c| c.num_gates())
    }
}

fn taso(
    circ: &Hugr,
    rewriter: &impl Rewriter,
    strategy: &impl RewriteStrategy,
    cost: impl Fn(&Hugr) -> usize,
    mut log_config: LogConfig,
    timeout: Option<u64>,
) -> Hugr {
    let start_time = Instant::now();

    let mut log_candidates = log_config.circ_candidates_csv.map(csv::Writer::from_writer);

    let mut best_circ = circ.clone();
    let mut best_circ_cost = cost(circ);
    log_best(best_circ_cost, log_candidates.as_mut()).unwrap();

    // Hash of seen circuits. Dot not store circuits as this map gets huge
    let mut seen_hashes: FxHashSet<_> = FromIterator::from_iter([(circ.circuit_hash())]);

    // The priority queue of circuits to be processed (this should not get big)
    let mut pq = HugrPQ::new(&cost);

    pq.push(circ.clone());

    let mut circ_cnt = 0;
    while let Some(Entry { circ, cost, .. }) = pq.pop() {
        if cost < best_circ_cost {
            best_circ = circ.clone();
            best_circ_cost = cost;
            log_best(best_circ_cost, log_candidates.as_mut()).unwrap();
        }

        let rewrites = rewriter.get_rewrites(&circ);
        for new_circ in strategy.apply_rewrites(rewrites, &circ) {
            let new_circ_hash = new_circ.circuit_hash();
            circ_cnt += 1;
            if circ_cnt % 1000 == 0 {
                log_progress(
                    log_config.progress_log.as_mut(),
                    circ_cnt,
                    &pq,
                    &seen_hashes,
                )
                .expect("Failed to write to progress log");
            }
            if seen_hashes.contains(&new_circ_hash) {
                continue;
            }
            pq.push_with_hash_unchecked(new_circ, new_circ_hash);
            seen_hashes.insert(new_circ_hash);
        }

        if pq.len() >= 10000 {
            // Haircut to keep the queue size manageable
            pq.truncate(5000);
        }

        if let Some(timeout) = timeout {
            if start_time.elapsed().as_secs() > timeout {
                println!("Timeout");
                break;
            }
        }
    }

    log_final(
        &best_circ,
        log_config.progress_log.as_mut(),
        log_config.final_circ_json.as_mut(),
        &cost,
    )
    .expect("Failed to write to progress log and/or final circuit JSON");

    best_circ
}

/// Run the TASO optimiser on a circuit.
///
/// The optimiser will repeatedly rewrite the circuit using the rewriter and
/// the rewrite strategy, optimising the circuit according to the cost function
/// provided. Optionally, a timeout (in seconds) can be provided.
///
/// A log of the successive best candidate circuits can be found in the file
/// `best_circs.csv`. In addition, the final best circuit is retrievable in the
/// files `final_best_circ.gv` and `final_best_circ.json`.
///
/// This is the multi-threaded version of the optimiser. See [`taso`] for the
/// single-threaded version.
// TODO Support MPSC and expose in API
#[allow(dead_code)]
fn taso_mpsc(
    circ: Hugr,
    rewriter: impl Rewriter + Send + Clone + 'static,
    strategy: impl RewriteStrategy + Send + Clone + 'static,
    cost: impl Fn(&Hugr) -> usize + Send + Sync,
    log_config: LogConfig,
    timeout: Option<u64>,
    n_threads: usize,
) -> Hugr {
    let start_time = Instant::now();

    let mut log_candidates = log_config.circ_candidates_csv.map(csv::Writer::from_writer);

    println!("Spinning up {n_threads} threads");

    // channel for sending circuits from threads back to main
    let (t_main, r_main) = mpsc::sync_channel(n_threads * 100);

    let mut best_circ = circ.clone();
    let mut best_circ_cost = cost(&best_circ);
    let circ_hash = circ.circuit_hash();
    log_best(best_circ_cost, log_candidates.as_mut()).unwrap();

    // Hash of seen circuits. Dot not store circuits as this map gets huge
    let mut seen_hashes: FxHashSet<_> = FromIterator::from_iter([(circ_hash)]);

    // The priority queue of circuits to be processed (this should not get big)
    let mut pq = HugrPQ::new(&cost);
    pq.push(circ);

    // each thread scans for rewrites using all the patterns and
    // sends rewritten circuits back to main
    let (joins, threads_tx, signal_new_data): (Vec<_>, Vec<_>, Vec<_>) = (0..n_threads)
        .map(|_| spawn_pattern_matching_thread(t_main.clone(), rewriter.clone(), strategy.clone()))
        .multiunzip();

    let mut cycle_inds = (0..n_threads).cycle();
    let mut threads_empty = vec![true; n_threads];

    let mut circ_cnt = 0;
    loop {
        // Fill each thread workqueue with data from pq
        while let Some(Entry {
            circ,
            cost: &cost,
            hash,
        }) = pq.peek()
        {
            if cost < best_circ_cost {
                best_circ = circ.clone();
                best_circ_cost = cost;
                log_best(best_circ_cost, log_candidates.as_mut()).unwrap();
                // Now we only care about smaller circuits
                seen_hashes.clear();
                seen_hashes.insert(hash);
            }
            // try to send to first available thread
            // TODO: Consider using crossbeam-channel
            if let Some(next_ind) = cycle_inds.by_ref().take(n_threads).find(|next_ind| {
                let tx = &threads_tx[*next_ind];
                tx.try_send(Some(circ.clone())).is_ok()
            }) {
                pq.pop();
                // Unblock thread if waiting
                let _ = signal_new_data[next_ind].try_recv();
                threads_empty[next_ind] = false;
            } else {
                // All send channels are full, continue
                break;
            }
        }

        // Receive data from threads, add to pq
        // We compute the hashes in the threads because it's expensive
        while let Ok(received) = r_main.try_recv() {
            let Some((circ_hash, circ)) = received else {
                panic!("A thread panicked");
            };
            circ_cnt += 1;
            if circ_cnt % 1000 == 0 {
                println!("{circ_cnt} circuits...");
                println!("Queue size: {} circuits", pq.len());
                println!("Total seen: {} circuits", seen_hashes.len());
            }
            if seen_hashes.contains(&circ_hash) {
                continue;
            }
            pq.push_with_hash_unchecked(circ, circ_hash);
            seen_hashes.insert(circ_hash);
        }

        // Check if all threads are waiting for new data
        for (is_waiting, is_empty) in signal_new_data.iter().zip(threads_empty.iter_mut()) {
            if is_waiting.try_recv().is_ok() {
                *is_empty = true;
            }
        }
        // If everyone is waiting and we do not have new data, we are done
        if pq.is_empty() && threads_empty.iter().all(|&x| x) {
            break;
        }
        if let Some(timeout) = timeout {
            if start_time.elapsed().as_secs() > timeout {
                println!("Timeout");
                break;
            }
        }
        if pq.len() >= 10000 {
            // Haircut to keep the queue size manageable
            pq.truncate(5000);
        }
    }

    println!("Tried {circ_cnt} circuits");
    println!("Joining");

    for (join, tx, data_tx) in izip!(joins, threads_tx, signal_new_data) {
        // tell all the threads we're done and join the threads
        tx.send(None).unwrap();
        let _ = data_tx.try_recv();
        join.join().unwrap();
    }

    println!("END RESULT: {}", cost(&best_circ));
    fs::write("final_best_circ.gv", best_circ.dot_string()).unwrap();
    fs::write(
        "final_best_circ.json",
        serde_json::to_vec(&best_circ).unwrap(),
    )
    .unwrap();
    best_circ
}

fn spawn_pattern_matching_thread(
    tx_main: SyncSender<Option<(u64, Hugr)>>,
    rewriter: impl Rewriter + Send + 'static,
    strategy: impl RewriteStrategy + Send + 'static,
) -> (JoinHandle<()>, SyncSender<Option<Hugr>>, Receiver<()>) {
    // channel for sending circuits to each thread
    let (tx_thread, rx) = mpsc::sync_channel(1000);
    // A flag to wait until new data
    let (wait_new_data, signal_new_data) = mpsc::sync_channel(0);

    let jn = thread::spawn(move || {
        loop {
            if let Ok(received) = rx.try_recv() {
                let Some(sent_hugr): Option<Hugr> = received else {
                    // Terminate thread
                    break;
                };
                let rewrites = rewriter.get_rewrites(&sent_hugr);
                for new_circ in strategy.apply_rewrites(rewrites, &sent_hugr) {
                    let new_circ_hash = new_circ.circuit_hash();
                    tx_main.send(Some((new_circ_hash, new_circ))).unwrap();
                }
            } else {
                // We are out of work, wait for new data
                wait_new_data.send(()).unwrap();
            }
        }
    });

    (jn, tx_thread, signal_new_data)
}

/// A helper struct for logging improvements in circuit size seen during the
/// TASO execution.
//
// TODO: Replace this fixed logging. Report back intermediate results.
#[derive(serde::Serialize, Debug)]
struct BestCircSer {
    circ_len: usize,
    time: String,
}

impl BestCircSer {
    fn new(circ_len: usize) -> Self {
        let time = chrono::Local::now().to_rfc3339();
        Self { circ_len, time }
    }
}

fn log_best<W: io::Write>(cbest: usize, wtr: Option<&mut csv::Writer<W>>) -> io::Result<()> {
    let Some(wtr) = wtr else {
        return Ok(());
    };
    println!("new best of size {}", cbest);
    wtr.serialize(BestCircSer::new(cbest)).unwrap();
    wtr.flush()
}

fn log_progress<W: io::Write, P: Ord, C>(
    wr: Option<&mut W>,
    circ_cnt: usize,
    pq: &HugrPQ<P, C>,
    seen_hashes: &FxHashSet<u64>,
) -> io::Result<()> {
    if let Some(wr) = wr {
        writeln!(wr, "{circ_cnt} circuits...")?;
        writeln!(wr, "Queue size: {} circuits", pq.len())?;
        writeln!(wr, "Total seen: {} circuits", seen_hashes.len())?;
    }
    Ok(())
}

fn log_final<W1: io::Write, W2: io::Write>(
    best_circ: &Hugr,
    log: Option<&mut W1>,
    final_circ: Option<&mut W2>,
    cost: impl Fn(&Hugr) -> usize,
) -> io::Result<()> {
    if let Some(log) = log {
        writeln!(log, "END RESULT: {}", cost(best_circ))?;
    }
    if let Some(circ_writer) = final_circ {
        save_tk1_json_writer(best_circ, circ_writer).unwrap();
    }
    Ok(())
}
