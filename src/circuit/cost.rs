//! Cost definitions for a circuit.

use derive_more::From;
use hugr::ops::OpType;
use std::fmt::Debug;
use std::iter::Sum;
use std::num::NonZeroUsize;
use std::ops::Add;

use crate::ops::op_matches;
use crate::T2Op;

/// The cost for a group of operations in a circuit, each with cost `OpCost`.
pub trait CircuitCost: Add<Output = Self> + Sum<Self> + Debug + Default + Clone + Ord {
    /// Returns true if the cost is above the threshold.
    fn check_threshold(self, threshold: Self) -> bool;

    /// Divide the cost, rounded up.
    fn div_cost(self, n: NonZeroUsize) -> Self;
}

/// A pair of major and minor cost.
///
/// This is used to order circuits based on major cost first, then minor cost.
/// A typical example would be CX count as major cost and total gate count as
/// minor cost.
#[derive(Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, From)]
pub struct MajorMinorCost {
    major: usize,
    minor: usize,
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

impl Debug for MajorMinorCost {
    // TODO: A nicer print for the logs
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "(major={}, minor={})", self.major, self.minor)
    }
}

impl Add<MajorMinorCost> for MajorMinorCost {
    type Output = MajorMinorCost;

    fn add(self, rhs: MajorMinorCost) -> Self::Output {
        (self.major + rhs.major, self.minor + rhs.minor).into()
    }
}

impl Sum for MajorMinorCost {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.reduce(|a, b| (a.major + b.major, a.minor + b.minor).into())
            .unwrap_or_default()
    }
}

impl CircuitCost for MajorMinorCost {
    #[inline]
    fn check_threshold(self, threshold: Self) -> bool {
        self.major > threshold.major
    }

    #[inline]
    fn div_cost(mut self, n: NonZeroUsize) -> Self {
        self.major = (self.major.saturating_sub(1)) / n.get() + 1;
        self.minor = (self.minor.saturating_sub(1)) / n.get() + 1;
        self
    }
}

impl CircuitCost for usize {
    #[inline]
    fn check_threshold(self, threshold: Self) -> bool {
        self > threshold
    }

    #[inline]
    fn div_cost(self, n: NonZeroUsize) -> Self {
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
