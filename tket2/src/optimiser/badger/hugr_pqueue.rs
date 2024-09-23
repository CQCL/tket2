use delegate::delegate;
use fxhash::FxHashMap;
use priority_queue::DoublePriorityQueue;

/// A min-priority queue for Hugrs.
///
/// The cost function provided will be used as the priority of the Hugrs.
/// Uses hashes internally to store the Hugrs.
#[derive(Debug, Clone, Default)]
pub struct HugrPQ<Circuit, P: Ord, HashFn> {
    queue: DoublePriorityQueue<u64, P>,
    hash_lookup: FxHashMap<u64, Circuit>,
    hash_fn: HashFn,
    max_size: usize,
}

pub struct Entry<C, P> {
    pub circ: C,
    pub cost: P,
}

impl<Circuit, P: Ord, HashFn> HugrPQ<Circuit, P, HashFn> {
    /// Create a new HugrPQ with a cost function and some initial capacity.
    pub fn new(max_size: usize, hash_fn: HashFn) -> Self {
        Self {
            queue: DoublePriorityQueue::with_capacity(max_size),
            hash_lookup: Default::default(),
            max_size,
            hash_fn,
        }
    }

    /// Reference to the minimal circuit in the queue.
    #[allow(unused)]
    pub fn peek(&self) -> Option<Entry<&Circuit, &P>> {
        let (hash, cost) = self.queue.peek_min()?;
        let circ = self.hash_lookup.get(hash)?;
        Some(Entry { circ, cost })
    }

    /// Push a circuit into the queue.
    ///
    /// If the queue is full, the element with the highest cost will be dropped.
    #[allow(unused)]
    pub fn push(&mut self, circ: Circuit, cost: P)
    where
        HashFn: Fn(&Circuit) -> u64,
    {
        let hash = (self.hash_fn)(&circ);
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
    pub fn push_unchecked(&mut self, circ: Circuit, hash: u64, cost: P) {
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
    pub fn pop(&mut self) -> Option<Entry<Circuit, P>> {
        let (hash, cost) = self.queue.pop_min()?;
        let circ = self.hash_lookup.remove(&hash)?;
        Some(Entry { circ, cost })
    }

    /// Pop the maximal circuit from the queue.
    pub fn pop_max(&mut self) -> Option<Entry<Circuit, P>> {
        let (hash, cost) = self.queue.pop_max()?;
        let circ = self.hash_lookup.remove(&hash)?;
        Some(Entry { circ, cost })
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
