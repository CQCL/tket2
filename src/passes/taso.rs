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
mod qtz_circuit;

use delegate::delegate;
use fxhash::{FxHashMap, FxHashSet};
use hugr::{Hugr, HugrView};
use itertools::{izip, Itertools};
use priority_queue::DoublePriorityQueue;
use std::sync::mpsc::{self, Receiver, SyncSender};
use std::thread::{self, JoinHandle};
use std::time::Instant;
use std::{fs, io};
use std::{fs::File, path::Path};

use serde::{Deserialize, Serialize};

use crate::circuit::CircuitHash;
use crate::rewrite::strategy::RewriteStrategy;
use crate::rewrite::Rewriter;

/// A set of circuits forming an Equivalence Circuit Class (ECC).
///
/// The set contains a distinguished circuit called the representative circuit.
/// By convention, this will be the first circuit in a collection. For rewriting
/// purposes, it is typically chosen to be the smallest.
#[derive(Clone, Serialize, Deserialize)]
pub struct EqCircClass {
    rep_circ: Hugr,
    /// Other equivalent circuits to the representative.
    other_circs: Vec<Hugr>,
}

impl EqCircClass {
    /// Create a new equivalence class with a representative circuit.
    pub fn new(rep_circ: Hugr, other_circs: Vec<Hugr>) -> Self {
        Self {
            rep_circ,
            other_circs,
        }
    }

    /// The representative circuit of the equivalence class.
    pub fn rep_circ(&self) -> &Hugr {
        &self.rep_circ
    }

    /// The other circuits in the equivalence class.
    pub fn others(&self) -> &[Hugr] {
        &self.other_circs
    }

    /// All circuits in the equivalence class.
    pub fn circuits(&self) -> impl Iterator<Item = &Hugr> {
        std::iter::once(&self.rep_circ).chain(self.other_circs.iter())
    }

    /// Consume into circuits of the equivalence class.
    pub fn into_circuits(self) -> impl Iterator<Item = Hugr> {
        std::iter::once(self.rep_circ).chain(self.other_circs)
    }

    /// The number of circuits in the equivalence class.
    ///
    /// An ECC always has a representative circuit, so this method will always
    /// return an integer strictly greater than 0.
    pub fn n_circuits(&self) -> usize {
        self.other_circs.len() + 1
    }
}

impl FromIterator<Hugr> for EqCircClass {
    fn from_iter<T: IntoIterator<Item = Hugr>>(iter: T) -> Self {
        let mut iter = iter.into_iter();
        let rep_circ = iter.next().unwrap();
        let other_circs = iter.collect();
        Self::new(rep_circ, other_circs)
    }
}

/// Load a set of equivalence classes from a JSON file.
pub fn load_eccs_json_file(path: impl AsRef<Path>) -> Vec<EqCircClass> {
    let all_circs = qtz_circuit::load_ecc_set(path);

    all_circs
        .into_values()
        .map(FromIterator::from_iter)
        .collect()
}

/// A min-priority queue for Hugrs.
///
/// The cost function provided will be used as the priority of the Hugrs.
/// Uses hashes internally to store the Hugrs.
#[derive(Debug, Clone, Default)]
struct HugrPQ<P: Ord, C> {
    queue: DoublePriorityQueue<u64, P>,
    hash_lookup: FxHashMap<u64, Hugr>,
    cost_fn: C,
}

struct Entry<C, P, H> {
    circ: C,
    cost: P,
    hash: H,
}

impl<P: Ord, C> HugrPQ<P, C> {
    /// Create a new HugrPQ with a cost function.
    fn new(cost_fn: C) -> Self {
        Self {
            queue: DoublePriorityQueue::new(),
            hash_lookup: Default::default(),
            cost_fn,
        }
    }

    /// Reference to the minimal Hugr in the queue.
    fn peek(&self) -> Option<Entry<&Hugr, &P, u64>> {
        let (hash, cost) = self.queue.peek_min()?;
        let circ = self.hash_lookup.get(hash)?;
        Some(Entry {
            circ,
            cost,
            hash: *hash,
        })
    }

    /// Push a Hugr into the queue.
    fn push(&mut self, hugr: Hugr)
    where
        C: Fn(&Hugr) -> P,
    {
        let hash = hugr.circuit_hash();
        self.push_with_hash_unchecked(hugr, hash);
    }

    /// Push a Hugr into the queue with a precomputed hash.
    ///
    /// This is useful to avoid recomputing the hash in [`HugrPQ::push`] when
    /// it is already known.
    ///
    /// This does not check that the hash is valid.
    fn push_with_hash_unchecked(&mut self, hugr: Hugr, hash: u64)
    where
        C: Fn(&Hugr) -> P,
    {
        let cost = (self.cost_fn)(&hugr);
        self.queue.push(hash, cost);
        self.hash_lookup.insert(hash, hugr);
    }

    /// Pop the minimal Hugr from the queue.
    fn pop(&mut self) -> Option<Entry<Hugr, P, u64>> {
        let (hash, cost) = self.queue.pop_min()?;
        let circ = self.hash_lookup.remove(&hash)?;
        Some(Entry { circ, cost, hash })
    }

    /// Discard the largest elements of the queue.
    ///
    /// Only keep up to `max_size` elements.
    fn truncate(&mut self, max_size: usize) {
        while self.queue.len() > max_size {
            self.queue.pop_max();
        }
    }

    delegate! {
        to self.queue {
            fn len(&self) -> usize;
            fn is_empty(&self) -> bool;
        }
    }
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
/// This is the single-threaded version of the optimiser. See [`taso_mpsc`] for
/// the multi-threaded version.
pub fn taso(
    circ: Hugr,
    rewriter: impl Rewriter,
    strategy: impl RewriteStrategy,
    cost: impl Fn(&Hugr) -> usize,
    timeout: Option<u64>,
) -> Hugr {
    let start_time = Instant::now();

    let file = File::create("best_circs.csv").unwrap();
    let mut log_best_circ = csv::Writer::from_writer(file);

    let mut best_circ = circ.clone();
    let mut best_circ_cost = cost(&circ);
    log_best(best_circ_cost, &mut log_best_circ).unwrap();

    // Hash of seen circuits. Dot not store circuits as this map gets huge
    let mut seen_hashes: FxHashSet<_> = FromIterator::from_iter([(circ.circuit_hash())]);

    // The priority queue of circuits to be processed (this should not get big)
    let mut pq = HugrPQ::new(&cost);

    pq.push(circ);

    let mut circ_cnt = 0;
    while let Some(Entry { circ, cost, hash }) = pq.pop() {
        if cost < best_circ_cost {
            best_circ = circ.clone();
            best_circ_cost = cost;
            log_best(best_circ_cost, &mut log_best_circ).unwrap();
            // Now we only care about smaller circuits
            seen_hashes.clear();
            seen_hashes.insert(hash);
        }

        let rewrites = rewriter.get_rewrites(&circ);
        for new_circ in strategy.apply_rewrites(rewrites, &circ) {
            let new_circ_hash = new_circ.circuit_hash();
            circ_cnt += 1;
            if circ_cnt % 1000 == 0 {
                println!("{circ_cnt} circuits...");
                println!("Queue size: {} circuits", pq.len());
                println!("Total seen: {} circuits", seen_hashes.len());
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

    println!("END RESULT: {}", cost(&best_circ));
    fs::write("final_best_circ.gv", best_circ.dot_string()).unwrap();
    fs::write(
        "final_best_circ.json",
        serde_json::to_vec(&best_circ).unwrap(),
    )
    .unwrap();
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
// TODO refactor so both implementations share more code
pub fn taso_mpsc(
    circ: Hugr,
    rewriter: impl Rewriter + Send + Clone + 'static,
    strategy: impl RewriteStrategy + Send + Clone + 'static,
    cost: impl Fn(&Hugr) -> usize + Send + Sync,
    timeout: Option<u64>,
    n_threads: usize,
) -> Hugr {
    let start_time = Instant::now();

    let file = File::create("best_circs.csv").unwrap();
    let mut log_best_circ = csv::Writer::from_writer(file);

    println!("Spinning up {n_threads} threads");

    // channel for sending circuits from threads back to main
    let (t_main, r_main) = mpsc::sync_channel(n_threads * 100);

    let mut best_circ = circ.clone();
    let mut best_circ_cost = cost(&best_circ);
    let circ_hash = circ.circuit_hash();
    log_best(best_circ_cost, &mut log_best_circ).unwrap();

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
        // Send data in pq to the threads
        while let Some(Entry {
            circ,
            cost: &cost,
            hash,
        }) = pq.peek()
        {
            if cost < best_circ_cost {
                best_circ = circ.clone();
                best_circ_cost = cost;
                log_best(best_circ_cost, &mut log_best_circ).unwrap();
                // Now we only care about smaller circuits
                seen_hashes.clear();
                seen_hashes.insert(hash);
            }
            // try to send to first available thread
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

fn log_best(cbest: usize, wtr: &mut csv::Writer<File>) -> io::Result<()> {
    println!("new best of size {}", cbest);
    wtr.serialize(BestCircSer::new(cbest)).unwrap();
    wtr.flush()
}
