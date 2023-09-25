use std::collections::VecDeque;

use fxhash::FxHashSet;

/// A datastructure storing Hugr hashes.
///
/// Stores hashes in buckets based on a cost function, to allow clearing
/// the set of hashes that are no longer needed.
pub(super) struct HugrHashSet {
    buckets: VecDeque<FxHashSet<u64>>,
    /// The cost at the front of the queue.
    min_cost: Option<usize>,
}

impl HugrHashSet {
    /// Create a new empty set.
    pub(super) fn new() -> Self {
        Self {
            buckets: VecDeque::new(),
            min_cost: None,
        }
    }

    /// Create a new set with a single hash and cost.
    pub(super) fn singleton(hash: u64, cost: usize) -> Self {
        let mut set = Self::new();
        set.insert(hash, cost);
        set
    }

    /// Insert circuit with given hash and cost.
    ///
    /// Returns whether the insertion was successful, i.e. the negation
    /// of whether it was already present.
    pub(super) fn insert(&mut self, hash: u64, cost: usize) -> bool {
        let Some(min_cost) = self.min_cost.as_mut() else {
            self.min_cost = Some(cost);
            self.buckets.push_front([hash].into_iter().collect());
            return true;
        };
        while cost < *min_cost {
            self.buckets.push_front(FxHashSet::default());
            *min_cost -= 1;
        }
        let bucket_index = cost - *min_cost;
        while bucket_index >= self.buckets.len() {
            self.buckets.push_back(FxHashSet::default());
        }
        self.buckets[bucket_index].insert(hash)
    }

    /// Returns whether the given hash is present in the set.
    pub(super) fn contains(&self, hash: u64, cost: usize) -> bool {
        let Some(min_cost) = self.min_cost else {
            return false;
        };
        let Some(index) = cost.checked_sub(min_cost) else {
            return false;
        };
        let Some(b) = self.buckets.get(index) else {
            return false;
        };
        b.contains(&hash)
    }

    fn max_cost(&self) -> Option<usize> {
        Some(self.min_cost? + self.buckets.len() - 1)
    }

    /// Remove all hashes with cost strictly greater than the given cost.
    pub(super) fn clear_over(&mut self, cost: usize) {
        while self.max_cost().is_some() && self.max_cost() > Some(cost) {
            self.buckets.pop_back();
            if self.buckets.is_empty() {
                self.min_cost = None;
            }
        }
    }

    /// The number of hashes in the set
    pub(super) fn len(&self) -> usize {
        self.buckets.iter().map(|b| b.len()).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::HugrHashSet;

    #[test]
    fn insert_elements() {
        // For simplicity, we use as cost: hash % 10
        let mut set = HugrHashSet::new();

        assert!(!set.contains(0, 0));
        assert!(!set.contains(2, 0));
        assert!(!set.contains(2, 3));

        assert!(set.insert(20, 2));
        assert!(!set.contains(0, 0));
        assert!(!set.insert(20, 2));
        assert!(set.insert(22, 2));
        assert!(set.insert(23, 2));

        assert!(set.contains(22, 2));

        assert!(set.insert(33, 3));
        assert_eq!(set.min_cost, Some(2));
        assert_eq!(set.max_cost(), Some(3));
        assert_eq!(
            set.buckets,
            [
                [20, 22, 23].into_iter().collect(),
                [33].into_iter().collect()
            ]
        );

        assert!(set.insert(3, 0));
        assert!(set.insert(1, 0));
        assert!(!set.insert(22, 2));
        assert!(set.contains(33, 3));
        assert!(set.contains(3, 0));
        assert!(!set.contains(3, 2));
        assert_eq!(set.min_cost, Some(0));
        assert_eq!(set.max_cost(), Some(3));

        assert_eq!(set.min_cost, Some(0));
        assert_eq!(
            set.buckets,
            [
                [1, 3].into_iter().collect(),
                [].into_iter().collect(),
                [20, 22, 23].into_iter().collect(),
                [33].into_iter().collect(),
            ]
        );
    }

    #[test]
    fn remove_empty() {
        let mut set = HugrHashSet::new();
        assert!(set.insert(20, 2));
        assert!(set.insert(30, 3));

        assert_eq!(set.len(), 2);
        set.clear_over(2);
        assert_eq!(set.len(), 1);
        set.clear_over(0);
        assert_eq!(set.len(), 0);
        assert!(set.min_cost.is_none());
    }
}
