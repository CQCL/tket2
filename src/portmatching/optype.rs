//! Subsets of `Hugr::OpType`s used for pattern matching.
//!
//! The main reason we cannot support the full HUGR set is because
//! some custom or black box optypes are not comparable and hashable.
//!
//! We currently support the minimum set of operations needed
//! for circuit pattern matching.

use std::{hash::Hash, num::NonZeroU8};

use hugr::ops::{LeafOp, OpType};

/// A subset of LeafOp for pattern matching.
///
/// Currently supporting: H, T, S, X, Y, Z, Tadj, Sadj, CX, ZZMax, Measure,
/// RzF64, Xor.
///
/// Using non-supported [`LeafOp`] variants will result in "Unsupported LeafOp"
/// panics.
#[derive(Clone, Debug, Eq)]
pub struct MatchLeafOp(LeafOp);

impl MatchLeafOp {
    fn ind(&self) -> Option<NonZeroU8> {
        match self.0 {
            LeafOp::H => NonZeroU8::new(1),
            LeafOp::T => NonZeroU8::new(2),
            LeafOp::S => NonZeroU8::new(3),
            LeafOp::X => NonZeroU8::new(4),
            LeafOp::Y => NonZeroU8::new(5),
            LeafOp::Z => NonZeroU8::new(6),
            LeafOp::Tadj => NonZeroU8::new(7),
            LeafOp::Sadj => NonZeroU8::new(8),
            LeafOp::CX => NonZeroU8::new(9),
            LeafOp::ZZMax => NonZeroU8::new(10),
            LeafOp::Measure => NonZeroU8::new(11),
            LeafOp::RzF64 => NonZeroU8::new(12),
            LeafOp::Xor => NonZeroU8::new(13),
            _ => None,
        }
    }

    fn ind_unchecked(&self) -> u8 {
        self.ind().expect("Unsupported LeafOp").get()
    }
}

impl PartialEq for MatchLeafOp {
    fn eq(&self, other: &Self) -> bool {
        self.ind_unchecked() == other.ind_unchecked()
    }
}

impl Hash for MatchLeafOp {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        state.write_u8(self.ind_unchecked())
    }
}

impl PartialOrd for MatchLeafOp {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.ind_unchecked().partial_cmp(&other.ind_unchecked())
    }
}

impl Ord for MatchLeafOp {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.ind().cmp(&other.ind())
    }
}

impl TryFrom<LeafOp> for MatchLeafOp {
    type Error = &'static str;

    fn try_from(value: LeafOp) -> Result<Self, Self::Error> {
        let value = MatchLeafOp(value);
        value.ind().ok_or("Unsupported LeafOp")?;
        Ok(value)
    }
}

/// A subset of `Hugr::OpType`s for pattern matching.
///
/// Currently supporting: Input, Output, LeafOp, LoadConstant.
#[derive(Clone, Debug, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub enum MatchOp {
    Input,
    Output,
    LeafOp(MatchLeafOp),
    LoadConstant,
}

impl TryFrom<OpType> for MatchOp {
    type Error = &'static str;

    fn try_from(value: OpType) -> Result<Self, Self::Error> {
        match value {
            OpType::Input(_) => Ok(MatchOp::Input),
            OpType::Output(_) => Ok(MatchOp::Output),
            OpType::LeafOp(op) => Ok(MatchOp::LeafOp(MatchLeafOp::try_from(op)?)),
            OpType::LoadConstant(_) => Ok(MatchOp::LoadConstant),
            _ => Err("Unsupported OpType"),
        }
    }
}
