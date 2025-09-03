//! Modify nodes related to function calls.

use hugr::{
    builder::Dataflow,
    core::HugrNode,
    hugr::hugrmut::HugrMut,
    ops::{Call, CallIndirect, LoadFunction, OpType},
};

use super::{ModifierError, ModifierResolver, ModifierResolverErrors};

impl<N: HugrNode> ModifierResolver<N> {
    pub(super) fn modify_call(
        &mut self,
        h: &impl HugrMut<Node = N>,
        n: N,
        call: &Call,
        new_dfg: &mut impl Dataflow,
    ) -> Result<(), ModifierResolverErrors<N>> {
        // TODO
        Ok(())
    }

    pub(super) fn modify_indirect_call(
        &mut self,
        h: &impl HugrMut<Node = N>,
        n: N,
        indir_call: &CallIndirect,
        new_dfg: &mut impl Dataflow,
    ) -> Result<(), ModifierResolverErrors<N>> {
        // TODO
        // self.add_node_no_modification(new_dfg, OpType::CallIndirect(indir_call), h, n)?;
        Ok(())
    }

    pub(super) fn modify_load_function(
        &mut self,
        h: &impl HugrMut<Node = N>,
        n: N,
        load: &LoadFunction,
        new_dfg: &mut impl Dataflow,
    ) -> Result<(), ModifierResolverErrors<N>> {
        // TODO
        self.add_node_no_modification(new_dfg, OpType::LoadFunction(load.clone()), h, n)?;
        Ok(())
    }
}
