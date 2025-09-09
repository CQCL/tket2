//! Pattern matching in circuits.

use std::cmp;

use derive_more::derive::From;
use hugr::{
    extension::simple_op::MakeExtensionOp,
    ops::{ExtensionOp, OpType},
    Direction, HugrView, Port,
};

use crate::{
    resource::{CircuitUnit, ResourceId, ResourceScope},
    TketOp,
};

use super::Subcircuit;

mod hugr_adapter;
pub use hugr_adapter::{HugrMatchAdapter, MatchingOptions};
#[cfg(feature = "badgerv2_unstable")]
mod im_adapter;
#[cfg(feature = "badgerv2_unstable")]
pub use im_adapter::ImMatchAdapter;
#[cfg(feature = "badgerv2_unstable")]
pub(crate) use im_adapter::ImMatchResult;

/// The result of extending a match to a new Tket operation.
///
/// Any combination of the following outcomes is possible:
/// - A complete match: the op was successfully matched, so the matching
///   subcircuit should be extended accordingly. The resulting match will be
///   reported back to the user and pattern matching will stop.
/// - A partial match: the op was successfully matched, so the matching
///   subcircuit should be extended accordingly. The pattern matching will
///   proceed and the partial match will not be reported to the user (until a
///   complete match is found).
/// - A skip: the current op should not be matched but matching should proceed
///   without it.
///
/// If none of the three outcomes is reported, pattern matching will stop
/// without reporting any matches.
#[derive(Debug, Clone)]
pub struct MatchOutcome<PartialMatchInfo, MatchInfo> {
    /// A complete match: add current op to the match and report it to the user.
    pub complete: Option<MatchInfo>,
    /// A partial match: add current op to the match and continue matching.
    pub proceed: Option<Update<PartialMatchInfo>>,
    /// Skip the current op and continue matching.
    pub skip: Option<Update<PartialMatchInfo>>,
}

/// Whether a value is updated. If it is, the new value is provided.
///
/// Isomorphic to `Option<T>`.
#[derive(Debug, Clone, Default, From)]
pub enum Update<T> {
    /// The value is unchanged.
    #[default]
    Unchanged,
    /// The value is changed.
    Changed(#[from] T),
}

impl<T> Update<T> {
    /// Map the update value.
    pub fn map<U>(self, f: impl FnOnce(T) -> U) -> Update<U> {
        match self {
            Update::Unchanged => Update::Unchanged,
            Update::Changed(x) => Update::Changed(f(x)),
        }
    }

    /// Apply the update to the old value, either returning the unchanged value
    /// or the new value.
    pub fn updated(self, old_val: &T) -> T
    where
        T: Clone,
    {
        match self {
            Update::Unchanged => old_val.clone(),
            Update::Changed(x) => x,
        }
    }
}

impl<T> From<Option<T>> for Update<T> {
    fn from(value: Option<T>) -> Self {
        match value {
            Some(x) => Update::Changed(x),
            None => Update::Unchanged,
        }
    }
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
    pub fn skip(self, partial_match: impl Into<Update<PartialMatchInfo>>) -> Self {
        Self {
            skip: Some(partial_match.into()),
            ..self
        }
    }

    /// Report a complete match: add current op to the match and report it to
    /// the user.
    #[must_use]
    pub fn complete(self, match_info: MatchInfo) -> Self {
        Self {
            complete: Some(match_info),
            ..self
        }
    }

    /// Report a partial match: add current op to the match and continue
    /// matching.
    #[must_use]
    pub fn proceed(self, partial_match: impl Into<Update<PartialMatchInfo>>) -> Self {
        Self {
            proceed: Some(partial_match.into()),
            ..self
        }
    }

    /// Internal utility to iterate over the match outcomes.
    #[must_use]
    fn into_enum_vec(
        self,
        curr_partial_match: &PartialMatchInfo,
    ) -> Vec<MatchOutcomeEnum<PartialMatchInfo, MatchInfo>>
    where
        PartialMatchInfo: Clone,
    {
        let mut out = Vec::new();
        if let Some(complete) = self.complete {
            out.push(MatchOutcomeEnum::Complete(complete));
        }
        if let Some(proceed) = self.proceed {
            out.push(MatchOutcomeEnum::Proceed(
                proceed.updated(curr_partial_match),
            ));
        }
        if let Some(skip) = self.skip {
            out.push(MatchOutcomeEnum::Skip(skip.updated(curr_partial_match)));
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
            proceed: self.proceed.map(|x| x.map(&f)),
            skip: self.skip.map(|x| x.map(&f)),
        }
    }
}

/// Context of a partial match, passed to [`CircuitMatcher::match_tket_op`].
#[derive(Clone)]
pub struct MatchContext<'c, PartialMatchInfo, H: HugrView> {
    /// The current partial match.
    pub subcircuit: &'c Subcircuit<H::Node>,
    /// The current partial match info.
    pub match_info: PartialMatchInfo,
    /// The circuit containing the current partial match.
    pub circuit: &'c ResourceScope<H>,
    /// The node of the current operation.
    pub op_node: H::Node,
}

impl<'c, PartialMatchInfo, H: HugrView> MatchContext<'c, PartialMatchInfo, H> {
    /// Whether the current op is before, after or within the matched subcircuit
    /// on the given resource.
    pub fn op_position(&self, resource: ResourceId) -> Option<cmp::Ordering> {
        let interval = self.subcircuit.get_interval(resource)?;
        Some(interval.position_in_interval(self.circuit.get_position(self.op_node)?))
    }
}

/// A trait for pattern matching in circuits.
pub trait CircuitMatcher {
    /// Context of a partial match.
    type PartialMatchInfo: Clone + Default;

    /// Description of a full match.
    type MatchInfo;

    /// Whether to match a (quantum) operation and expand the match.
    fn match_tket_op<H: HugrView>(
        &self,
        op: TketOp,
        args: &[CircuitUnit<H::Node>],
        match_context: MatchContext<Self::PartialMatchInfo, H>,
    ) -> MatchOutcome<Self::PartialMatchInfo, Self::MatchInfo>;

    /// Whether to match an opaque operation and expand the match.
    ///
    /// Defaults to ignoring opaque extension operations.
    fn match_extension_op<H: HugrView>(
        &self,
        _op: ExtensionOp,
        _inputs: &[CircuitUnit<H::Node>],
        _outputs: &[CircuitUnit<H::Node>],
        _match_context: MatchContext<Self::PartialMatchInfo, H>,
    ) -> MatchOutcome<Self::PartialMatchInfo, Self::MatchInfo> {
        MatchOutcome::default().skip(None)
    }

    /// Convert the matcher to a [`HugrMatchAdapter`], specialised in matching
    /// patterns in concrete circuits.
    fn as_hugr_matcher(&self) -> HugrMatchAdapter<'_, Self> {
        HugrMatchAdapter { matcher: self }
    }

    /// Convert the matcher to a [`ImMatchAdapter`], specialised in matching
    /// patterns in [`RewriteSpace`]s.
    ///
    /// [`RewriteSpace`]: crate::rewrite_space::RewriteSpace
    #[cfg(feature = "badgerv2_unstable")]
    fn as_rewrite_space_matcher(&self) -> ImMatchAdapter<'_, Self> {
        ImMatchAdapter { matcher: self }
    }
}

enum TketOrExtensionOp {
    Tket(TketOp),
    Extension(ExtensionOp),
    Unsupported,
}

fn as_tket_or_extension_op(op: &OpType) -> TketOrExtensionOp {
    let Some(ext_op) = op.as_extension_op() else {
        return TketOrExtensionOp::Unsupported;
    };
    if let Ok(tket_op) = TketOp::from_extension_op(ext_op) {
        TketOrExtensionOp::Tket(tket_op)
    } else {
        TketOrExtensionOp::Extension(ext_op.clone())
    }
}

/// Iterator over all ports of a node of linear type.
fn all_linear_ports<H: HugrView>(host: &H, node: H::Node) -> impl Iterator<Item = Port> + '_ {
    host.value_types(node, Direction::Incoming)
        .chain(host.value_types(node, Direction::Outgoing))
        .filter_map(|(port, typ)| (!typ.copyable()).then_some(port))
}

enum MatchOutcomeEnum<PartialMatchInfo, MatchInfo> {
    Complete(MatchInfo),
    Proceed(PartialMatchInfo),
    Skip(PartialMatchInfo),
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A matcher finding all sequences of two or more Hadamard gates.
    pub(crate) struct TestHadamardMatcher;
    type NumHadamards = usize;

    impl CircuitMatcher for TestHadamardMatcher {
        type PartialMatchInfo = NumHadamards;
        type MatchInfo = NumHadamards;

        fn match_tket_op<H: HugrView>(
            &self,
            op: TketOp,
            _args: &[CircuitUnit<H::Node>],
            match_context: MatchContext<Self::PartialMatchInfo, H>,
        ) -> MatchOutcome<Self::PartialMatchInfo, Self::MatchInfo> {
            // We can always skip this op
            let mut outcomes = MatchOutcome::default().skip(Update::Unchanged);
            match op {
                TketOp::H => {
                    // we have a hadamard, so we can match this op and proceed
                    let num_hadamards = match_context.match_info + 1;
                    outcomes = outcomes.proceed(num_hadamards);
                    if num_hadamards >= 2 {
                        // We have enough hadamards to report the current match
                        outcomes.complete(num_hadamards)
                    } else {
                        // Proceed (without reporting a match)
                        outcomes
                    }
                }
                _ => outcomes,
            }
        }
    }

    /// A matcher finding Rz gates with constant angle `0.123`.
    pub(crate) struct TestRzMatcher;

    impl CircuitMatcher for TestRzMatcher {
        type PartialMatchInfo = ();
        type MatchInfo = ();

        fn match_tket_op<H: HugrView>(
            &self,
            op: TketOp,
            args: &[CircuitUnit<H::Node>],
            match_context: MatchContext<Self::PartialMatchInfo, H>,
        ) -> MatchOutcome<Self::PartialMatchInfo, Self::MatchInfo> {
            if op == TketOp::Rz && match_context.circuit.as_const_f64(args[1]) == Some(0.123) {
                MatchOutcome::default().complete(())
            } else {
                MatchOutcome::stop()
            }
        }
    }
}
