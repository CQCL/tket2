//! Cost definitions for a circuit.

use derive_more::From;
use hugr::ops::OpType;
use std::fmt::{Debug, Display};
use std::iter::Sum;
use std::num::NonZeroUsize;
use std::ops::Add;

use crate::ops::op_matches;
use crate::T2Op;

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
pub trait CostDelta: Sum<Self> + Debug + Default + Clone + Ord {
    /// Return the delta as a `isize`. This may discard some of the cost delta information.
    fn as_isize(&self) -> isize;
}

/// A pair of major and minor cost.
///
/// This is used to order circuits based on major cost first, then minor cost.
/// A typical example would be CX count as major cost and total gate count as
/// minor cost.
#[derive(Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, From)]
pub struct MajorMinorCost<T = usize> {
    major: T,
    minor: T,
}

// Serialise as string so that it is easy to write to CSV
impl serde::Serialize for MajorMinorCost {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_str(&format!("{:?}", self))
    }
}

impl<T: Display> Debug for MajorMinorCost<T> {
    // TODO: A nicer print for the logs
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "(major={}, minor={})", self.major, self.minor)
    }
}

impl Add for MajorMinorCost {
    type Output = MajorMinorCost;

    fn add(self, rhs: MajorMinorCost) -> Self::Output {
        (self.major + rhs.major, self.minor + rhs.minor).into()
    }
}

impl<T: Add<Output = T> + Default> Sum for MajorMinorCost<T> {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.reduce(|a, b| (a.major + b.major, a.minor + b.minor).into())
            .unwrap_or_default()
    }
}

impl CostDelta for MajorMinorCost<isize> {
    #[inline]
    fn as_isize(&self) -> isize {
        self.major
    }
}

impl CircuitCost for MajorMinorCost<usize> {
    type CostDelta = MajorMinorCost<isize>;

    #[inline]
    fn as_usize(&self) -> usize {
        self.major
    }

    #[inline]
    fn sub_cost(&self, other: &Self) -> Self::CostDelta {
        let major = (self.major as isize) - (other.major as isize);
        let minor = (self.minor as isize) - (other.minor as isize);
        MajorMinorCost { major, minor }
    }

    #[inline]
    fn add_delta(&self, delta: &Self::CostDelta) -> Self {
        MajorMinorCost {
            major: self.major.saturating_add_signed(delta.major),
            minor: self.minor.saturating_add_signed(delta.minor),
        }
    }

    #[inline]
    fn div_cost(&self, n: NonZeroUsize) -> Self {
        let major = (self.major.saturating_sub(1)) / n.get() + 1;
        let minor = (self.minor.saturating_sub(1)) / n.get() + 1;
        Self { major, minor }
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
    op_matches(op, T2Op::CX)
}

/// Returns true if the operation is a quantum operation.
pub fn is_quantum(op: &OpType) -> bool {
    let Ok(op): Result<T2Op, _> = op.try_into() else {
        return false;
    };
    op.is_quantum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn major_minor() {
        let a = MajorMinorCost {
            major: 10,
            minor: 2,
        };
        let b = MajorMinorCost {
            major: 20,
            minor: 1,
        };
        assert!(a < b);
        assert_eq!(
            a + b,
            MajorMinorCost {
                major: 30,
                minor: 3
            }
        );
        assert_eq!(a.sub_cost(&b).as_isize(), -10);
        assert_eq!(b.sub_cost(&a).as_isize(), 10);
        assert_eq!(
            a.div_cost(NonZeroUsize::new(2).unwrap()),
            MajorMinorCost { major: 5, minor: 1 }
        );
        assert_eq!(
            a.div_cost(NonZeroUsize::new(3).unwrap()),
            MajorMinorCost { major: 4, minor: 1 }
        );
        assert_eq!(
            a.div_cost(NonZeroUsize::new(1).unwrap()),
            MajorMinorCost {
                major: 10,
                minor: 2
            }
        );
    }
}
