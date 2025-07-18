//! A priority queue for states, with support for maximum size and considering
//! every state at most once.

use delegate::delegate;
use fxhash::{FxHashMap, FxHashSet};
use priority_queue::DoublePriorityQueue;

use super::State;

/// A min-priority queue for states in the search space.
///
/// The states are ordered according to their cost. Uses hashes internally to
/// store the states.
///
/// States can be inserted at most once. Any further insertions will fail.
#[derive(Debug, Clone, Default)]
pub(super) struct StatePQueue<S, P: Ord> {
    queue: DoublePriorityQueue<u64, P>,
    hash_lookup: FxHashMap<u64, S>,
    seen_hashes: FxHashSet<u64>,
    max_size: usize,
}

/// An entry in the priority queue.
pub(super) struct Entry<S, P, H> {
    /// The state.
    pub state: S,
    /// The cost of the state.
    pub cost: P,
    /// The hash of the state.
    pub hash: H,
}

impl<S, P: Ord> StatePQueue<S, P> {
    /// Create a new HugrPQ with a cost function and some initial capacity.
    pub fn new(max_size: usize) -> Self {
        Self {
            queue: DoublePriorityQueue::with_capacity(max_size),
            hash_lookup: Default::default(),
            seen_hashes: Default::default(),
            max_size,
        }
    }

    /// Reference to the minimal circuit in the queue.
    #[allow(unused)]
    pub fn peek(&self) -> Option<Entry<&S, &P, u64>> {
        let (&hash, cost) = self.queue.peek_min()?;
        let state = self.hash_lookup.get(&hash)?;
        Some(Entry { state, cost, hash })
    }

    /// Push into the queue.
    ///
    /// If the queue is full, the element with the highest cost will be dropped.
    ///
    /// Return the entry that was just inserted. If one of:
    ///  - hashing or costing failed,
    ///  - the priority queue was full and the cost higher than the highest cost
    ///    in the queue, or
    ///  - the state has already been seen,
    ///
    /// return `None` and the state will be discarded.
    pub fn push<Ctx>(&mut self, state: S, context: &Ctx) -> Option<Entry<&S, &P, u64>>
    where
        S: State<Ctx, Cost = P>,
    {
        let hash = state.hash(context)?;
        let cost = state.cost(context)?;
        self.push_unchecked(state, hash, cost)
    }

    pub fn push_unchecked(&mut self, state: S, hash: u64, cost: P) -> Option<Entry<&S, &P, u64>> {
        if !self.check_accepted(&cost) || !self.seen_hashes.insert(hash) {
            return None;
        }
        if self.len() >= self.max_size {
            self.pop_max();
        }
        self.queue.push(hash, cost);
        let (_, cost_ref) = self.queue.get(&hash).expect("just inserted");
        let state_ref = self.hash_lookup.entry(hash).or_insert(state);

        Some(Entry {
            state: state_ref,
            cost: cost_ref,
            hash,
        })
    }

    /// Pop the minimal state from the queue.
    pub fn pop(&mut self) -> Option<Entry<S, P, u64>> {
        let (hash, cost) = self.queue.pop_min()?;
        let state = self.hash_lookup.remove(&hash)?;
        Some(Entry { state, cost, hash })
    }

    /// Pop the maximal state from the queue.
    pub fn pop_max(&mut self) -> Option<Entry<S, P, u64>> {
        let (hash, cost) = self.queue.pop_max()?;
        let state = self.hash_lookup.remove(&hash)?;
        Some(Entry { state, cost, hash })
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

    /// The largest cost in the queue.
    pub fn max_cost(&self) -> Option<&P> {
        self.queue.peek_max().map(|(_, cost)| cost)
    }

    /// The number of unique hashes in the queue.
    pub fn num_seen_hashes(&self) -> usize {
        self.seen_hashes.len()
    }

    /// Returns `true` if an element with the given cost would be accepted.
    ///
    /// If `false`, the element will be dropped if passed to
    /// [`StatePQueue::push`].
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
