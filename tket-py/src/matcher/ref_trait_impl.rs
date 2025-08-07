use hugr::{hugr::views::SiblingSubgraph, HugrView};
// Note: These imports may need to be adjusted based on the actual tket crate structure
use tket::{
    rewrite::{
        matcher::{CircuitMatcher, MatchContext, MatchOutcome, OpArg},
        replacer::CircuitReplacer,
    },
    Circuit, TketOp,
};

/// A wrapper that implements a trait for a reference to a type that implements the trait.
///
/// Implemented for [`CircuitMatcher`] and [`CircuitReplacer`].
pub(super) struct RefTraitImpl<'a, T: ?Sized>(pub(super) &'a T);

impl<'a, T: CircuitMatcher + ?Sized> CircuitMatcher for RefTraitImpl<'a, T> {
    type PartialMatchInfo = T::PartialMatchInfo;

    type MatchInfo = T::MatchInfo;

    fn match_tket_op(
        &self,
        op: TketOp,
        op_args: &[OpArg],
        match_context: MatchContext<Self::PartialMatchInfo, impl HugrView>,
    ) -> MatchOutcome<Self::PartialMatchInfo, Self::MatchInfo> {
        self.0.match_tket_op(op, op_args, match_context)
    }
}

impl<'a, MatchInfo, T: CircuitReplacer<MatchInfo> + ?Sized> CircuitReplacer<MatchInfo>
    for RefTraitImpl<'a, T>
{
    fn replace_match<H: HugrView>(
        &self,
        subgraph: &SiblingSubgraph<H::Node>,
        hugr: H,
        match_info: MatchInfo,
    ) -> Vec<Circuit> {
        self.0.replace_match(subgraph, hugr, match_info)
    }
}
