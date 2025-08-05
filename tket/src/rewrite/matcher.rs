//! Pattern matching in circuits.

use hugr::{extension::simple_op::MakeExtensionOp, ops::OpType, HugrView};
use portgraph::algorithms::convex::LineIndex;

use crate::subcircuit::LineConvexChecker;
use crate::TketOp;

use super::Subcircuit;

mod adapter;
pub use adapter::{HugrMatchAdapter, MatchingOptions};

/// A type alias for the line partition index of a qubit in a circuit.
pub type QubitIndex = LineIndex;

/// An argument to a Tket operation.
///
/// Currently only support arguments that are of qubit type that are "pass-through",
/// i.e. are both the i-th input and the i-th output, as well as constant
/// rotation angles.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum OpArg {
    /// Argument is on the provided qubit index.
    ///
    /// The exact position of the operation along the qubit line is unspecified.
    Qubit(QubitIndex),
    /// Argument is on the provided qubit index, positioned before the first
    /// current operation on that qubit (with respect to a given subcircuit).
    QubitOpBefore(QubitIndex),
    /// Argument is on the provided qubit index, positioned after the last
    /// current operation on that qubit (with respect to a given subcircuit).
    QubitOpAfter(QubitIndex),
    /// Argument is a constant floating point value.
    ConstF64(f64),
}

impl OpArg {
    fn relative_to<H: HugrView>(
        self,
        node: H::Node,
        subcircuit: &Subcircuit,
        checker: &LineConvexChecker<H>,
    ) -> Option<OpArg> {
        let line = match self {
            OpArg::Qubit(line) => line,
            OpArg::QubitOpBefore(line) => line,
            OpArg::QubitOpAfter(line) => line,
            const_f64 @ OpArg::ConstF64(_) => return Some(const_f64),
        };

        let pos = checker.get_position(node);
        let Some(interval) = subcircuit.get_interval(line) else {
            // Unknown qubit line, cannot specify relative position.
            return Some(OpArg::Qubit(line));
        };

        if pos < interval.min {
            return Some(OpArg::QubitOpBefore(line));
        } else if pos > interval.max {
            return Some(OpArg::QubitOpAfter(line));
        }

        // Invalid relative position: neither before nor after subcircuit.
        None
    }
}

/// The result of extending a match to a new Tket operation.
///
/// Any combination of the following outcomes is possible:
/// - A complete match: the op was successfully matched, so the matching subcircuit
///   should be extended accordingly. The resulting match will be reported back
///   to the user and pattern matching will stop.
/// - A partial match: the op was successfully matched, so the matching subcircuit
///   should be extended accordingly. The pattern matching will proceed and the
///   partial match will not be reported to the user (until a complete match is found).
/// - A skip: the current op should not be matched but matching should proceed
///   without it.
///
/// If none of the three outcomes is reported, pattern matching will stop without
/// reporting any matches.
#[derive(Debug, Clone)]
pub struct MatchOutcome<PartialMatchInfo, MatchInfo> {
    /// A complete match: add current op to the match and report it to the user.
    complete: Option<MatchInfo>,
    /// A partial match: add current op to the match and continue matching.
    proceed: Option<PartialMatchInfo>,
    /// Skip the current op and continue matching.
    skip: Option<PartialMatchInfo>,
}

impl<PartialMatchInfo, MatchInfo> Default for MatchOutcome<PartialMatchInfo, MatchInfo> {
    fn default() -> Self {
        Self {
            complete: None,
            proceed: None,
            skip: None,
        }
    }
}

impl<PartialMatchInfo, MatchInfo> MatchOutcome<PartialMatchInfo, MatchInfo> {
    /// Stop matching.
    #[must_use]
    pub fn stop() -> Self {
        Self::default()
    }

    /// Skip the current op and continue matching.
    #[must_use]
    pub fn skip(self, partial_match: PartialMatchInfo) -> Self {
        Self {
            skip: Some(partial_match),
            ..self
        }
    }

    /// Report a complete match: add current op to the match and report it to the user.
    #[must_use]
    pub fn complete(self, match_info: MatchInfo) -> Self {
        Self {
            complete: Some(match_info),
            ..self
        }
    }

    /// Report a partial match: add current op to the match and continue matching.
    #[must_use]
    pub fn proceed(self, partial_match: PartialMatchInfo) -> Self {
        Self {
            proceed: Some(partial_match),
            ..self
        }
    }

    /// Internal utility to iterate over the match outcomes.
    #[must_use]
    fn into_enum_vec(self) -> Vec<MatchOutcomeEnum<PartialMatchInfo, MatchInfo>> {
        let mut out = Vec::new();
        if let Some(complete) = self.complete {
            out.push(MatchOutcomeEnum::Complete(complete));
        }
        if let Some(proceed) = self.proceed {
            out.push(MatchOutcomeEnum::Proceed(proceed));
        }
        if let Some(skip) = self.skip {
            out.push(MatchOutcomeEnum::Skip(skip));
        }
        out
    }

    /// Map the match outcomes to new types.
    #[must_use]
    pub fn map<NewPartialMatchInfo, NewMatchInfo>(
        self,
        f: impl Fn(PartialMatchInfo) -> NewPartialMatchInfo,
        g: impl FnOnce(MatchInfo) -> NewMatchInfo,
    ) -> MatchOutcome<NewPartialMatchInfo, NewMatchInfo> {
        MatchOutcome {
            complete: self.complete.map(g),
            proceed: self.proceed.map(&f),
            skip: self.skip.map(f),
        }
    }
}

/// Context of a partial match, passed to [`CircuitMatcher::match_tket_op`].
#[derive(Clone)]
pub struct MatchContext<'c, 'g, PartialMatchInfo, H: HugrView> {
    /// The current partial match.
    pub subcircuit: &'c Subcircuit,
    /// The current partial match info.
    pub match_info: PartialMatchInfo,
    /// The convex checker for the current partial match.
    pub checker: &'c LineConvexChecker<'g, H>,
    /// The node of the current operation.
    pub op_node: H::Node,
}

/// A trait for pattern matching in circuits.
pub trait CircuitMatcher {
    /// Context of a partial match.
    type PartialMatchInfo: Clone + Default;

    /// Description of a full match.
    type MatchInfo;

    /// Whether to match a (quantum) operation and expand the match.
    fn match_tket_op(
        &self,
        op: TketOp,
        op_args: &[OpArg],
        match_context: MatchContext<Self::PartialMatchInfo, impl HugrView>,
    ) -> MatchOutcome<Self::PartialMatchInfo, Self::MatchInfo>;

    /// Convert the matcher to a [`HugrMatchAdapter`], specialised in matching
    /// patterns in concrete circuits.
    fn as_hugr_matcher(&self) -> HugrMatchAdapter<Self> {
        HugrMatchAdapter { matcher: self }
    }
}

fn as_tket_op(op: &OpType) -> Option<TketOp> {
    let ext_op = op.as_extension_op()?;
    TketOp::from_extension_op(ext_op).ok()
}

enum MatchOutcomeEnum<PartialMatchInfo, MatchInfo> {
    Complete(MatchInfo),
    Proceed(PartialMatchInfo),
    Skip(PartialMatchInfo),
}
