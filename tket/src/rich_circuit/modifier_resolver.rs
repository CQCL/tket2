//! Try to delete modifier by applying the modifier to each component.
//!
//! For function calls, the modifier is

use std::{collections::HashMap, mem::swap};

use derive_more::Error;
use hugr::{
    core::HugrNode,
    extension::simple_op::{MakeExtensionOp, MakeOpDef},
    hugr::{
        hugrmut::HugrMut,
        patch::{replace::ReplaceError, PatchHugrMut, PatchVerification},
    },
    ops::{DataflowOpTrait, ExtensionOp, OpType},
    types::{FuncTypeBase, TypeBase},
    HugrView, Node,
};
use itertools::Itertools;

use crate::{
    rich_circuit::{CombinedModifier, Modifier},
    Circuit, Tk2Op,
};

/// Error that can occur when resolving modifiers.
#[derive(Debug, Error, derive_more::Display)]
pub enum ModifierError<N = Node> {
    /// The node is not a modifier
    #[display("Node to modify {_0} expected to be a modifier but actually {_1}")]
    NotModifier(N, OpType),
    /// The node cannot be modified.
    #[display("Modification by {_0:?} is not defined for the node {_1}")]
    Unimplemented(Modifier, OpType),
    /// No caller of this modified function exists.
    #[display("No caller of the modified function exists for node {_0}")]
    NoCaller(N),
    /// No target of this modifer exists.
    #[display("No caller of the modified function exists for node {_0}")]
    NoTarget(N),
    /// Not the first modifier in a chain.
    #[display("Node {_0} is not the first modifier in a chain. It is called by {_0}")]
    NotInitialModifier(N, OpType),
    /// Modifier applied to a node that cannot be modified.
    #[display("Modifier {_0} cannot be applied to the node {_1}")]
    ModifierNotApplicable(N, OpType),
}

#[derive(Debug, derive_more::Display)]
enum ModifierResolverErrors<N = Node> {
    ReplaceError(ReplaceError),
    ModifierError(ModifierError<N>),
}
impl<N> From<ModifierError<N>> for ModifierResolverErrors<N> {
    fn from(err: ModifierError<N>) -> Self {
        ModifierResolverErrors::ModifierError(err)
    }
}
impl<N> From<ReplaceError> for ModifierResolverErrors<N> {
    fn from(err: ReplaceError) -> Self {
        ModifierResolverErrors::ReplaceError(err)
    }
}

pub struct ModifierResolver<N = Node> {
    node: N,
    // TODO:
    // Should keep track of the collection of modifiers that are applied to the same function.
    // This will prevent the duplicated generation of Controlled-functions.
}

impl<N: HugrNode> ModifierResolver<N> {
    fn verify(&self, h: &impl HugrView<Node = N>) -> Result<(), ModifierError<N>> {
        // Check if the node is a modifier, modifying an operation.
        let optype = h.get_optype(self.node);
        if Modifier::from_optype(optype).is_none() {
            return Err(ModifierError::NotModifier(self.node, optype.clone()));
        }
        // Check if this is the first modifier in a chain.
        // FIXME: I assume there is exactly one edge goint out of the modifier.
        // This sounds true in most cases, but may not be guaranteed.
        if let Some((caller, _)) = h.all_linked_outputs(self.node).exactly_one().ok() {
            let optype = h.get_optype(caller);
            if Modifier::from_optype(optype).is_some() {
                return Err(ModifierError::NotInitialModifier(caller, optype.clone()));
            }
        } else {
            return Err(ModifierError::NoCaller(self.node));
        }
        Ok(())
    }

    fn try_rewrite(self, h: &mut impl HugrMut<Node = N>) -> Result<(), ModifierResolverErrors<N>> {
        // Verify that the rewrite can be applied.
        self.verify(h)?;

        // The final target of modifiers to apply.
        let mut targ = self.node.clone();
        // Collection of modifiers to apply.
        let mut modifiers: Vec<Modifier> = Vec::new();
        loop {
            let optype = h.get_optype(targ);
            match Modifier::from_optype(optype) {
                Some(modifier) => modifiers.push(modifier),
                // Found the target
                None => break,
            }
            targ = h
                .all_linked_inputs(targ)
                .exactly_one()
                .ok()
                .map(|(n, _)| n)
                .ok_or(ModifierError::NoTarget(self.node))?;
        }

        // Calculate the accumulated modifier.
        let combined_modifier: CombinedModifier = modifiers.into();

        let optype = h.get_optype(targ);
        match optype {
            OpType::Input(_) => return Err(ModifierError::NoTarget(self.node).into()),
            OpType::DataflowBlock(d) => todo!(),
            OpType::LoadFunction(func) => todo!(),
            OpType::LoadConstant(const_func) => todo!(),
            // TODO: Handle modifiers for other op types.
            // For example, tail loops, or conditionals.
            _ => {
                return Err(ModifierError::ModifierNotApplicable(self.node, optype.clone()).into())
            }
        }

        Ok(())
    }
}

/// Resolve modifiers in a circuit by applying them to each entry point.
pub fn resolve_modifier_with_entrypoints(
    h: &mut impl HugrMut<Node = Node>,
    entry_points: impl Iterator<Item = Node>,
) -> Result<(), ReplaceError<Node>> {
    use ModifierResolverErrors::*;

    for entry_point in entry_points {
        let children = h.children(entry_point).collect::<Vec<_>>();
        for node in children {
            let resolver = ModifierResolver { node };
            // Verify the resolver can be applied.
            match resolver.try_rewrite(h) {
                Ok(_) => (),
                // If not resolvable, just skip.
                Err(ModifierError(_)) => continue,
                // ReplaceError will be actual error.
                Err(ReplaceError(e)) => return Err(e),
            }
        }
    }
    Ok(())
}
