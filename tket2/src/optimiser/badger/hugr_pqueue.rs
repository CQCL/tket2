use delegate::delegate;
use fxhash::FxHashMap;
use priority_queue::DoublePriorityQueue;

use crate::circuit::CircuitHash;
use crate::Circuit;

/// A min-priority queue for Hugrs.
///
/// The cost function provided will be used as the priority of the Hugrs.
/// Uses hashes internally to store the Hugrs.
#[derive(Debug, Clone, Default)]
pub struct HugrPQ<P: Ord, C> {
    queue: DoublePriorityQueue<u64, P>,
    hash_lookup: FxHashMap<u64, Circuit>,
    cost_fn: C,
    max_size: usize,
}

pub struct Entry<C, P, H> {
    pub circ: C,
    pub cost: P,
    pub hash: H,
}

impl<P: Ord, C> HugrPQ<P, C> {
    /// Create a new HugrPQ with a cost function and some initial capacity.
    pub fn new(cost_fn: C, max_size: usize) -> Self {
        Self {
            queue: DoublePriorityQueue::with_capacity(max_size),
            hash_lookup: Default::default(),
            cost_fn,
            max_size,
        }
    }

    /// Reference to the minimal circuit in the queue.
    #[allow(unused)]
    pub fn peek(&self) -> Option<Entry<&Circuit, &P, u64>> {
        let (hash, cost) = self.queue.peek_min()?;
        let circ = self.hash_lookup.get(hash)?;
        Some(Entry {
            circ,
            cost,
            hash: *hash,
        })
    }

    /// Push a circuit into the queue.
    ///
    /// If the queue is full, the element with the highest cost will be dropped.
    #[allow(unused)]
    pub fn push(&mut self, circ: Circuit)
    where
        C: Fn(&Circuit) -> P,
    {
        let hash = circ.circuit_hash().unwrap();
        let cost = (self.cost_fn)(&circ);
        self.push_unchecked(circ, hash, cost);
    }

    /// Push a circuit into the queue with a precomputed hash and cost.
    ///
    /// This is useful to avoid recomputing the hash and cost function in
    /// [`HugrPQ::push`] when they are already known.
    ///
    /// This does not check that the hash is valid.
    ///
    /// If the queue is full, the most last will be dropped.
    pub fn push_unchecked(&mut self, circ: Circuit, hash: u64, cost: P)
    where
        C: Fn(&Circuit) -> P,
    {
        if !self.check_accepted(&cost) {
            return;
        }
        if self.len() >= self.max_size {
            self.pop_max();
        }
        self.queue.push(hash, cost);
        self.hash_lookup.insert(hash, circ);
    }

    /// Pop the minimal circuit from the queue.
    pub fn pop(&mut self) -> Option<Entry<Circuit, P, u64>> {
        let (hash, cost) = self.queue.pop_min()?;
        let circ = self.hash_lookup.remove(&hash)?;
        Some(Entry { circ, cost, hash })
    }

    /// Pop the maximal circuit from the queue.
    pub fn pop_max(&mut self) -> Option<Entry<Circuit, P, u64>> {
        let (hash, cost) = self.queue.pop_max()?;
        let circ = self.hash_lookup.remove(&hash)?;
        Some(Entry { circ, cost, hash })
    }

    /// Discard the largest elements of the queue.
    ///
    /// Only keep up to `max_size` elements.
    #[allow(unused)]
    pub fn truncate(&mut self, max_size: usize) {
        while self.queue.len() > max_size {
            let (hash, _) = self.queue.pop_max().unwrap();
            self.hash_lookup.remove(&hash);
        }
    }

    /// The cost function used by the queue.
    #[allow(unused)]
    pub fn cost_fn(&self) -> &C {
        &self.cost_fn
    }

    /// The largest cost in the queue.
    pub fn max_cost(&self) -> Option<&P> {
        self.queue.peek_max().map(|(_, cost)| cost)
    }

    /// Returns `true` if an element with the given cost would be accepted.
    ///
    /// If `false`, the element will be dropped if passed to [`HugrPQ::push`] or
    /// [`HugrPQ::push_unchecked`].
    pub fn check_accepted(&self, cost: &P) -> bool {
        if self.max_size == 0 {
            return false;
        }
        if self.len() < self.max_size {
            return true;
        }
        cost < self.max_cost().unwrap()
    }

    /// Returns `true` is the queue is at capacity.
    pub fn is_full(&self) -> bool {
        self.queue.len() >= self.max_size
    }

    delegate! {
        to self.queue {
            pub fn len(&self) -> usize;
            pub fn is_empty(&self) -> bool;
        }
    }
}
