//! Modify nodes related to function calls.

use std::mem;

use chrono::offset;
use hugr::{
    builder::{BuildError, Dataflow},
    core::HugrNode,
    extension::simple_op::MakeExtensionOp,
    hugr::hugrmut::HugrMut,
    ops::{Call, CallIndirect, DataflowOpTrait, LoadFunction, OpType},
    types::EdgeKind,
    IncomingPort, OutgoingPort, Wire,
};
use itertools::Itertools;

use crate::rich_circuit::{
    modifier_resolver::{DirWire, PortExt},
    Modifier,
};

use super::{ModifierError, ModifierResolver, ModifierResolverErrors};

impl<N: HugrNode> ModifierResolver<N> {
    pub(super) fn modify_call(
        &mut self,
        h: &mut impl HugrMut<Node = N>,
        n: N,
        optype: &OpType,
        new_dfg: &mut impl Dataflow,
    ) -> Result<(), ModifierResolverErrors<N>> {
        let OpType::Call(call) = optype else {
            return Err(ModifierResolverErrors::Unreachable(
                "Not a Call".to_string(),
            ));
        };
        let offset = self.modifiers().accum_ctrl.len();

        let mut poly_sig = call.func_sig.clone();
        let type_args = call.type_args.clone();
        self.modify_signature(poly_sig.body_mut(), false);
        let new_call = Call::try_new(poly_sig, type_args).map_err(BuildError::from)?;
        let signature = (*new_call.signature()).clone();
        let new_call_fn_port = new_call.called_function_port();
        let new_call_node = new_dfg.add_child_node(new_call);
        let callee = h
            .single_linked_output(n, call.called_function_port())
            .unwrap();

        // wire the callee
        let new_callee = self.modify_fn(h, callee.0)?;
        self.call_map()
            .insert(new_callee, (new_call_node, new_call_fn_port));
        // wire the controls
        let mut controls = self.pack_controls(new_dfg)?;
        for (i, control) in controls.iter_mut().enumerate() {
            new_dfg
                .hugr_mut()
                .connect(control.node(), control.source(), new_call_node, i);
            *control = Wire::new(new_call_node, i);
        }
        let controls = self.unpack_controls(new_dfg, controls)?;
        *self.controls() = controls;
        // wire the inputs/outputs
        self.wire_node_inout(
            n,
            new_call_node,
            signature.input.iter(),
            signature.output.iter(),
            0,
            0,
            offset,
        )?;

        Ok(())
    }

    /// Apply the collected chain of modifiers to the function loaded by the `LoadFunction` node.
    /// Returns the new node that loads the modified function.
    /// This applies changes to the original graph `h`.
    pub(super) fn apply_modifier_chain_to_loaded_fn(
        &mut self,
        h: &mut impl HugrMut<Node = N>,
        n: N,
    ) -> Result<N, ModifierResolverErrors<N>> {
        // The final target of modifiers to apply.
        // Collection of modifiers to apply.
        let modifiers_and_targ = self.trace_modifiers_chain(h, n)?;
        let targ = modifiers_and_targ
            .last()
            .cloned()
            .ok_or(ModifierError::NoTarget(n))?;

        // The function to apply the modifier to.
        let (func, load) = Self::get_loaded_function(h, n, targ, h.get_optype(targ))?;

        // Modify the function
        let modified_fn = self.modify_fn(h, func)?;

        // Modify the function loader
        // Insert the new LoadFunction node to load the modified function
        let mut modified_sig = load.func_sig.clone();
        self.modify_signature(modified_sig.body_mut(), false);
        let load = LoadFunction::try_new(modified_sig, load.type_args).map_err(BuildError::from)?;
        let new_load = h.add_node_after(n, load);
        h.connect(modified_fn, 0, new_load, 0);

        // Delete the modifiers, and change the function to be loaded
        // FIXME: This could delete the node that is referenced by other nodes that is not in the chain.
        for mod_or_targ in modifiers_and_targ {
            // This should disconnect. Should not remove.
            // h.disconnect(n, port);
            // h.remove_node(mod_or_targ);
        }
        Ok(new_load)
    }

    /// Trace the chain of modifiers starting from node `n`, collecting all modifier nodes until reaching
    /// a non-modifier target node. Returns the chain of nodes in order from the starting node to the target node.
    /// The return includes the starting node and the target node.
    pub(super) fn trace_modifiers_chain(
        &mut self,
        h: &impl HugrMut<Node = N>,
        n: N,
    ) -> Result<Vec<N>, ModifierError<N>> {
        // The final target of modifiers to apply.
        let mut current = n;
        // Collection of modifiers to apply.
        let modifiers = self.modifiers_mut();
        let mut chain: Vec<N> = Vec::new();
        loop {
            chain.push(current);
            let optype = h.get_optype(current);

            if Modifier::from_optype(optype).is_none() {
                break;
            }

            modifiers.push(optype.as_extension_op().unwrap());
            let next = h
                .single_linked_output(current, 0)
                .ok_or(ModifierError::NoTarget(n))?;
            current = next.0;
        }
        Ok(chain)
    }

    /// Given a target node `targ` which is expected to be a `LoadFunction`, retrieve the function node it loads.
    pub(super) fn get_loaded_function<'a>(
        h: &impl HugrMut<Node = N>,
        n: N,
        targ: N,
        optype: &'a OpType,
    ) -> Result<(N, LoadFunction), ModifierError<N>> {
        match optype {
            OpType::LoadFunction(load) => {
                let (fn_node, _) = h.single_linked_output(targ, 0).unwrap();
                let fn_optype = h.get_optype(fn_node);
                let OpType::FuncDefn(_) = fn_optype else {
                    return Err({
                        ModifierError::ModifierNotApplicable(n, fn_optype.clone()).into()
                    });
                };
                // TODO: We want some machinery to prevent generating a lot of copies of modified functions
                // from the same function.
                Ok((fn_node, load.clone()))
            }
            OpType::Input(_) => return Err(ModifierError::NoTarget(n).into()),
            // If the target is a function, we need to create a new dataflow block of it.
            _ => {
                // TODO: Handle modifiers provided from other nodes.
                // For example, conditionals?
                return Err(ModifierError::ModifierNotApplicable(n, optype.clone()).into());
            }
        }
    }

    pub(super) fn modify_indirect_call(
        &mut self,
        h: &mut impl HugrMut<Node = N>,
        n: N,
        indir_call: &CallIndirect,
        new_dfg: &mut impl Dataflow,
    ) -> Result<(), ModifierResolverErrors<N>> {
        // Wrapper to convert ModifierError to UnResolvable with the indir_call node.
        // This is because, even if we find an error in the process immediately,
        // we cannot stop processing here.
        let wrap_err = |e: ModifierError<N>| {
            ModifierResolverErrors::UnResolvable(
                e.node(),
                "Cannot modify indirect call.".to_string(),
                indir_call.clone().into(),
            )
        };

        // Trace the chain of modifiers starting from the one before the indirect call.
        let chain_tail = h.single_linked_output(n, 0).unwrap();
        let modifiers = self.modifiers().clone();
        let trace = self
            .trace_modifiers_chain(h, chain_tail.0)
            .map_err(wrap_err)?;
        let targ = trace.last().cloned().unwrap();
        let (func, load) =
            Self::get_loaded_function(h, n, targ, h.get_optype(targ)).map_err(wrap_err)?;
        let modified_fn = self.modify_fn(h, func)?;

        // Make new LoadFunction
        let mut modified_sig = load.func_sig.clone();
        self.modify_signature(modified_sig.body_mut(), false);
        let load = LoadFunction::try_new(modified_sig, load.type_args).map_err(BuildError::from)?;
        let new_load = new_dfg.add_child_node(load);
        self.call_map()
            .insert(modified_fn, (new_load, IncomingPort::from(0)));
        *self.modifiers_mut() = modifiers; // TODO: Where should I restore modifiers?

        // Make new IndirectCall
        let mut new_call = indir_call.clone();
        self.modify_signature(&mut new_call.signature, false);
        let new_call_node = new_dfg.add_child_node(new_call);

        // Wire the new IndirectCall
        let mut controls = self.pack_controls(new_dfg)?;
        let offset = self.modifiers().accum_ctrl.len();
        for (i, ctrl) in controls.iter_mut().enumerate() {
            new_dfg
                .hugr_mut()
                .connect(ctrl.node(), ctrl.source(), new_call_node, i + 1);
            *ctrl = Wire::new(new_call_node, i);
        }
        *self.controls() = self.unpack_controls(new_dfg, controls)?;

        let signature = indir_call.signature();
        self.wire_node_inout(
            n,
            new_call_node,
            signature.input.iter().skip(1),
            signature.output.iter(),
            1,
            0,
            offset,
        )?;
        new_dfg.hugr_mut().connect(new_load, 0, new_call_node, 0);
        self.map_insert_none((n, IncomingPort::from(0)).into())?;

        // FIXME: Forgetting all the nodes in the chain so that we don't have to worry about mapping the edges.
        // Otherwise, there would be edges in the original graph that have no corresponding edges in the new graph.
        // However, this could remove wires referenced by other nodes that are not in the chain.
        for node in trace {
            self.forget_node(h, node)?
        }

        Ok(())
    }

    pub(super) fn modify_load_function(
        &mut self,
        _h: &impl HugrMut<Node = N>,
        _n: N,
        _load: &LoadFunction,
        _new_dfg: &mut impl Dataflow,
    ) -> Result<(), ModifierResolverErrors<N>> {
        // Indirect calles needs to be handled by its caller.
        // TODO: Might want to load function
        Ok(())
    }
}
