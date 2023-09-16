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
    cost_fn: C,
}

pub(super) struct Entry<C, P, H> {
    pub(super) circ: C,
    pub(super) cost: P,
    #[allow(unused)] // TODO remove?
    pub(super) hash: H,
}

impl<P: Ord, C> HugrPQ<P, C> {
    /// Create a new HugrPQ with a cost function and some initial capacity.
    pub(super) fn with_capacity(cost_fn: C, capacity: usize) -> Self {
        Self {
            queue: DoublePriorityQueue::with_capacity(capacity),
            hash_lookup: Default::default(),
            cost_fn,
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
    pub(super) fn push(&mut self, hugr: Hugr)
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
    pub(super) fn push_with_hash_unchecked(&mut self, hugr: Hugr, hash: u64)
    where
        C: Fn(&Hugr) -> P,
    {
        let cost = (self.cost_fn)(&hugr);
        self.queue.push(hash, cost);
        self.hash_lookup.insert(hash, hugr);
    }

    /// Pop the minimal Hugr from the queue.
    pub(super) fn pop(&mut self) -> Option<Entry<Hugr, P, u64>> {
        let (hash, cost) = self.queue.pop_min()?;
        let circ = self.hash_lookup.remove(&hash)?;
        Some(Entry { circ, cost, hash })
    }

    /// Discard the largest elements of the queue.
    ///
    /// Only keep up to `max_size` elements.
    pub(super) fn truncate(&mut self, max_size: usize) {
        while self.queue.len() > max_size {
            self.queue.pop_max();
        }
    }

    delegate! {
        to self.queue {
            pub(super) fn len(&self) -> usize;
            pub(super) fn is_empty(&self) -> bool;
        }
    }
}
