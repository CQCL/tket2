//! Subsets of `Hugr::OpType`s used for pattern matching.
//!
//! The main reason we cannot support the full HUGR set is because
//! some custom or black box optypes are not comparable and hashable.
//!
//! We currently support the minimum set of operations needed
//! for circuit pattern matching.

use std::hash::Hash;

use hugr::ops::{custom::ExternalOp, LeafOp, OpType};
use smol_str::SmolStr;

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
    fn id(&self) -> Option<SmolStr> {
        match &self.0 {
            // h_gate()
            // | LeafOp::T
            // | LeafOp::S
            // | LeafOp::X
            // | LeafOp::Y
            // | LeafOp::Z
            // | LeafOp::Tadj
            // | LeafOp::Sadj
            // | LeafOp::ZZMax
            // | LeafOp::Measure
            // | LeafOp::RzF64
            // | LeafOp::Xor => Some(self.0.name()),
            LeafOp::CustomOp(c) => match (*c).as_ref() {
                ExternalOp::Extension(e) => Some(e.def().name().clone()),
                _ => None,
            },
            _ => None,
        }
    }

    fn id_unchecked(&self) -> SmolStr {
        self.id().expect("Unsupported LeafOp")
    }
}

impl PartialEq for MatchLeafOp {
    fn eq(&self, other: &Self) -> bool {
        self.id_unchecked() == other.id_unchecked()
    }
}

impl Hash for MatchLeafOp {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.id_unchecked().hash(state)
    }
}

impl PartialOrd for MatchLeafOp {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.id_unchecked().partial_cmp(&other.id_unchecked())
    }
}

impl Ord for MatchLeafOp {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.id().cmp(&other.id())
    }
}

impl TryFrom<LeafOp> for MatchLeafOp {
    type Error = &'static str;

    fn try_from(value: LeafOp) -> Result<Self, Self::Error> {
        let value = MatchLeafOp(value);
        value.id().ok_or("Unsupported LeafOp")?;
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
