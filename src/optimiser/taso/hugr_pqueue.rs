use delegate::delegate;
use fxhash::FxHashMap;
use hugr::Hugr;
use priority_queue::DoublePriorityQueue;

use crate::circuit::CircuitHash;

/// A min-priority queue for Hugrs.
///
/// The cost function provided will be used as the priority of the Hugrs.
/// Uses hashes internally to store the Hugrs.
#[derive(Debug, Clone, Default)]
pub(super) struct HugrPQ<P: Ord, C> {
    queue: DoublePriorityQueue<u64, P>,
    hash_lookup: FxHashMap<u64, Hugr>,
    pub(super) cost_fn: C,
    max_size: usize,
}

pub(super) struct Entry<C, P, H> {
    pub(super) circ: C,
    pub(super) cost: P,
    #[allow(unused)] // TODO remove?
    pub(super) hash: H,
}

impl<P: Ord, C> HugrPQ<P, C> {
    /// Create a new HugrPQ with a cost function and some initial capacity.
    pub(super) fn new(cost_fn: C, max_size: usize) -> Self {
        Self {
            queue: DoublePriorityQueue::with_capacity(max_size),
            hash_lookup: Default::default(),
            cost_fn,
            max_size,
        }
    }

    /// Reference to the minimal Hugr in the queue.
    #[allow(unused)]
    pub(super) fn peek(&self) -> Option<Entry<&Hugr, &P, u64>> {
        let (hash, cost) = self.queue.peek_min()?;
        let circ = self.hash_lookup.get(hash)?;
        Some(Entry {
            circ,
            cost,
            hash: *hash,
        })
    }

    /// Push a Hugr into the queue.
    ///
    /// If the queue is full, the element with the highest cost will be dropped.
    pub(super) fn push(&mut self, hugr: Hugr)
    where
        C: Fn(&Hugr) -> P,
    {
        let hash = hugr.circuit_hash();
        let cost = (self.cost_fn)(&hugr);
        self.push_unchecked(hugr, hash, cost);
    }

    /// Push a Hugr into the queue with a precomputed hash and cost.
    ///
    /// This is useful to avoid recomputing the hash and cost function in
    /// [`HugrPQ::push`] when they are already known.
    ///
    /// This does not check that the hash is valid.
    ///
    /// If the queue is full, the most last will be dropped.
    pub(super) fn push_unchecked(&mut self, hugr: Hugr, hash: u64, cost: P)
    where
        C: Fn(&Hugr) -> P,
    {
        if self.max_size == 0 {
            return;
        }
        if self.len() >= self.max_size {
            if cost >= *self.max_cost().unwrap() {
                return;
            }
            self.pop_max();
        }
        self.queue.push(hash, cost);
        self.hash_lookup.insert(hash, hugr);
    }

    /// Pop the minimal Hugr from the queue.
    pub(super) fn pop(&mut self) -> Option<Entry<Hugr, P, u64>> {
        let (hash, cost) = self.queue.pop_min()?;
        let circ = self.hash_lookup.remove(&hash)?;
        Some(Entry { circ, cost, hash })
    }

    /// Pop the maximal Hugr from the queue.
    pub(super) fn pop_max(&mut self) -> Option<Entry<Hugr, P, u64>> {
        let (hash, cost) = self.queue.pop_max()?;
        let circ = self.hash_lookup.remove(&hash)?;
        Some(Entry { circ, cost, hash })
    }

    /// Discard the largest elements of the queue.
    ///
    /// Only keep up to `max_size` elements.
    pub(super) fn truncate(&mut self, max_size: usize) {
        while self.queue.len() > max_size {
            let (hash, _) = self.queue.pop_max().unwrap();
            self.hash_lookup.remove(&hash);
        }
    }

    /// The largest cost in the queue.
    #[allow(unused)]
    pub(super) fn max_cost(&self) -> Option<&P> {
        self.queue.peek_max().map(|(_, cost)| cost)
    }

    delegate! {
        to self.queue {
            pub(super) fn len(&self) -> usize;
            pub(super) fn is_empty(&self) -> bool;
        }
    }
}
