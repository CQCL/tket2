//! Cost definitions for a circuit.

use derive_more::From;
use hugr::ops::OpType;
use std::fmt::{Debug, Display};
use std::iter::Sum;
use std::num::NonZeroUsize;
use std::ops::{Add, AddAssign};

use crate::ops::op_matches;
use crate::Tk2Op;

/// The cost for a group of operations in a circuit, each with cost `OpCost`.
pub trait CircuitCost: Add<Output = Self> + Sum<Self> + Debug + Default + Clone + Ord {
    /// The cost delta between two costs.
    type CostDelta: CostDelta;

    /// Return the cost as a `usize`. This may discard some of the cost information.
    fn as_usize(&self) -> usize;

    /// Return the cost delta between two costs.
    fn sub_cost(&self, other: &Self) -> Self::CostDelta;

    /// Adds a cost delta to the cost.
    fn add_delta(&self, delta: &Self::CostDelta) -> Self;

    /// Divide the cost, rounded up.
    fn div_cost(&self, n: NonZeroUsize) -> Self;
}

/// The cost for a group of operations in a circuit, each with cost `OpCost`.
pub trait CostDelta:
    AddAssign + Add<Output = Self> + Sum<Self> + Debug + Default + Clone + Ord
{
    /// Return the delta as a `isize`. This may discard some of the cost delta information.
    fn as_isize(&self) -> isize;
}

/// A pair of major and minor cost.
///
/// This is used to order circuits based on major cost first, then minor cost.
/// A typical example would be CX count as major cost and total gate count as
/// minor cost.
pub type MajorMinorCost<T = usize> = LexicographicCost<T, 2>;

impl<T> From<(T, T)> for MajorMinorCost<T> {
    fn from((major, minor): (T, T)) -> Self {
        Self([major, minor])
    }
}

/// A cost that is ordered lexicographically.
///
/// An array of cost functions, where the first one is infinitely more important
/// than the second, which is infinitely more important than the third, etc.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, From)]
pub struct LexicographicCost<T, const N: usize>([T; N]);

impl<const N: usize, T: Default + Copy> Default for LexicographicCost<T, N> {
    fn default() -> Self {
        Self([Default::default(); N])
    }
}

// Serialise as string so that it is easy to write to CSV
impl<const N: usize> serde::Serialize for LexicographicCost<usize, N> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_str(&format!("{:?}", self))
    }
}

impl<T: Display, const N: usize> Debug for LexicographicCost<T, N> {
    // TODO: A nicer print for the logs
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}

impl<T: Add<Output = T> + Copy, const N: usize> Add for LexicographicCost<T, N> {
    type Output = Self;

    fn add(mut self, rhs: Self) -> Self::Output {
        for i in 0..N {
            self.0[i] = self.0[i] + rhs.0[i];
        }
        return self;
    }
}

impl<T: AddAssign + Copy, const N: usize> AddAssign for LexicographicCost<T, N> {
    fn add_assign(&mut self, rhs: Self) {
        for i in 0..N {
            self.0[i] += rhs.0[i];
        }
    }
}

impl<T: Add<Output = T> + Default + Copy, const N: usize> Sum for LexicographicCost<T, N> {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.reduce(|a, b| a + b).unwrap_or_default()
    }
}

impl<const N: usize> CostDelta for LexicographicCost<isize, N> {
    #[inline]
    fn as_isize(&self) -> isize {
        if N > 0 {
            self.0[0]
        } else {
            0
        }
    }
}

impl<const N: usize> CircuitCost for LexicographicCost<usize, N> {
    type CostDelta = LexicographicCost<isize, N>;

    #[inline]
    fn as_usize(&self) -> usize {
        if N > 0 {
            self.0[0]
        } else {
            0
        }
    }

    #[inline]
    fn sub_cost(&self, other: &Self) -> Self::CostDelta {
        let mut costdelta = [0; N];
        for i in 0..N {
            costdelta[i] = (self.0[i] as isize) - (other.0[i] as isize);
        }
        LexicographicCost(costdelta)
    }

    #[inline]
    fn add_delta(&self, delta: &Self::CostDelta) -> Self {
        let mut ret = [0; N];
        for i in 0..N {
            ret[i] = self.0[i].saturating_add_signed(delta.0[i]);
        }
        Self(ret)
    }

    #[inline]
    fn div_cost(&self, n: NonZeroUsize) -> Self {
        let mut ret = [0; N];
        for i in 0..N {
            ret[i] = (self.0[i].saturating_sub(1)) / n.get() + 1;
        }
        Self(ret)
    }
}

impl CostDelta for isize {
    #[inline]
    fn as_isize(&self) -> isize {
        *self
    }
}

impl CircuitCost for usize {
    type CostDelta = isize;

    #[inline]
    fn as_usize(&self) -> usize {
        *self
    }

    #[inline]
    fn sub_cost(&self, other: &Self) -> Self::CostDelta {
        (*self as isize) - (*other as isize)
    }

    #[inline]
    fn add_delta(&self, delta: &Self::CostDelta) -> Self {
        self.saturating_add_signed(*delta)
    }

    #[inline]
    fn div_cost(&self, n: NonZeroUsize) -> Self {
        (self.saturating_sub(1)) / n.get() + 1
    }
}

/// Returns true if the operation is a controlled X operation.
pub fn is_cx(op: &OpType) -> bool {
    op_matches(op, Tk2Op::CX)
}

/// Returns true if the operation is a quantum operation.
pub fn is_quantum(op: &OpType) -> bool {
    let Ok(op): Result<Tk2Op, _> = op.try_into() else {
        return false;
    };
    op.is_quantum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn major_minor() {
        let a = LexicographicCost([10, 2]);
        let b = LexicographicCost([20, 1]);
        assert!(a < b);
        assert_eq!(a + b, LexicographicCost([30, 3]));
        assert_eq!(a.sub_cost(&b).as_isize(), -10);
        assert_eq!(b.sub_cost(&a).as_isize(), 10);
        assert_eq!(
            a.div_cost(NonZeroUsize::new(2).unwrap()),
            LexicographicCost([5, 1])
        );
        assert_eq!(
            a.div_cost(NonZeroUsize::new(3).unwrap()),
            LexicographicCost([4, 1])
        );
        assert_eq!(
            a.div_cost(NonZeroUsize::new(1).unwrap()),
            LexicographicCost([10, 2])
        );
    }
}
