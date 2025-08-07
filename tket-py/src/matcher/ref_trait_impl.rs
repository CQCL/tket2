use hugr::{ops::ExtensionOp, HugrView};
// Note: These imports may need to be adjusted based on the actual tket crate
// structure
use tket::{
    resource::{CircuitUnit, ResourceScope},
    rewrite::{
        matcher::{CircuitMatcher, MatchContext, MatchOutcome},
        replacer::CircuitReplacer,
    },
    Circuit, Subcircuit, TketOp,
};

/// A wrapper that implements a trait for a reference to a type that implements
/// the trait.
///
/// Implemented for [`CircuitMatcher`] and [`CircuitReplacer`].
pub(super) struct RefTraitImpl<'a, T: ?Sized>(pub(super) &'a T);

impl<'a, T: CircuitMatcher + ?Sized> CircuitMatcher for RefTraitImpl<'a, T> {
    type PartialMatchInfo = T::PartialMatchInfo;

    type MatchInfo = T::MatchInfo;

    fn match_tket_op<H: HugrView>(
        &self,
        op: TketOp,
        args: &[CircuitUnit<H::Node>],
        match_context: MatchContext<Self::PartialMatchInfo, H>,
    ) -> MatchOutcome<Self::PartialMatchInfo, Self::MatchInfo> {
        self.0.match_tket_op(op, args, match_context)
    }

    fn match_extension_op<H: HugrView>(
        &self,
        op: ExtensionOp,
        inputs: &[CircuitUnit<H::Node>],
        outputs: &[CircuitUnit<H::Node>],
        match_context: MatchContext<Self::PartialMatchInfo, H>,
    ) -> MatchOutcome<Self::PartialMatchInfo, Self::MatchInfo> {
        self.0
            .match_extension_op(op, inputs, outputs, match_context)
    }
}

impl<'a, MatchInfo, T: CircuitReplacer<MatchInfo> + ?Sized> CircuitReplacer<MatchInfo>
    for RefTraitImpl<'a, T>
{
    fn replace_match<H: HugrView>(
        &self,
        subcircuit: &Subcircuit<H::Node>,
        circuit: &ResourceScope<H>,
        match_info: MatchInfo,
    ) -> Vec<Circuit> {
        self.0.replace_match(subcircuit, circuit, match_info)
    }
}
