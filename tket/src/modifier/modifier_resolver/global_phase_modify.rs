//! Modifier resolution for global phase operations.
//! After resolving modifiers, all global phase operations are removed.
//!
use hugr::{
    builder::Dataflow,
    core::HugrNode,
    extension::simple_op::MakeExtensionOp,
    hugr::hugrmut::HugrMut,
    std_extensions::arithmetic::{float_ops::FloatOps, float_types::ConstF64},
    IncomingPort, Node, Wire,
};

use crate::{
    extension::{global_phase::GlobalPhase, rotation::RotationOp},
    modifier::modifier_resolver::{connect, ModifierResolver, ModifierResolverErrors},
    TketOp,
};

impl<N: HugrNode> ModifierResolver<N> {
    /// Modify a global phase operation.
    /// This returns the incoming port for the rotation of the modified operation.
    pub fn modify_global_phase(
        &mut self,
        n: N,
        new_fn: &mut impl Dataflow,
        ancilla: &mut Vec<Wire<Node>>,
    ) -> Result<Vec<(Node, IncomingPort)>, ModifierResolverErrors<N>> {
        match (self.modifiers.dagger, self.control_num()) {
            (false, 0) => {
                let node = new_fn.add_child_node(GlobalPhase);
                let in_port = IncomingPort::from(0);
                Ok(vec![(node, in_port)])
            }
            (true, 0) => {
                let halfturn = new_fn.add_child_node(RotationOp::to_halfturns);
                let angle_float = Wire::new(halfturn, 0);
                let neg_angle_float = new_fn
                    .add_dataflow_op(FloatOps::fneg, vec![angle_float])
                    .map(|out| out.out_wire(0))?;
                let angle = new_fn
                    .add_dataflow_op(RotationOp::from_halfturns_unchecked, vec![neg_angle_float])
                    .map(|out| out.out_wire(0))?;
                new_fn.add_dataflow_op(GlobalPhase, vec![angle])?;
                Ok(vec![(halfturn, IncomingPort::from(0))])
            }
            // Cn+1Phase(cs, c, θ) = * CnPhase(cs, θ/2) * CnRz(cs, c, θ)
            (dagger, _) => {
                self.modifiers.dagger = false;

                let halfturn = new_fn.add_child_node(RotationOp::to_halfturns);
                let angle_float = Wire::new(halfturn, 0);

                let half = new_fn.add_load_value(ConstF64::new(if dagger { -0.5 } else { 0.5 }));
                let half_angle_float = new_fn
                    .add_dataflow_op(FloatOps::fmul, vec![angle_float, half])
                    .map(|out| out.out_wire(0))?;
                let angle_half = new_fn
                    .add_dataflow_op(RotationOp::from_halfturns_unchecked, vec![half_angle_float])
                    .map(|out| out.out_wire(0))?;

                let mut c = self.pop_control().unwrap();

                // C^nPhase(cs, θ/2)
                let c_phase = self.with_ancilla(&mut c, ancilla, |this, ancilla| {
                    this.modify_global_phase(n, new_fn, ancilla)
                })?;
                for (node, port) in c_phase {
                    new_fn
                        .hugr_mut()
                        .connect(angle_half.node(), angle_half.source(), node, port);
                }

                // C^nRz(cs, c, θ)
                let c_rz = self.modify_tket_op(n, TketOp::Rz, new_fn, ancilla)?;
                connect(new_fn, &c_rz.incoming[0], &c.into())?;
                c = c_rz.outgoing[0].clone().try_into().unwrap();

                let mut result = vec![(halfturn, IncomingPort::from(0))];

                if dagger {
                    let neg_angle_float = new_fn
                        .add_dataflow_op(FloatOps::fneg, vec![angle_float])
                        .map(|out| out.out_wire(0))?;
                    let angle = new_fn
                        .add_dataflow_op(
                            RotationOp::from_halfturns_unchecked,
                            vec![neg_angle_float],
                        )
                        .map(|out| out.out_wire(0))?;
                    connect(new_fn, &c_rz.incoming[1], &angle.into())?;
                } else {
                    let in_wire = c_rz.incoming[1].clone().try_into().unwrap();
                    result.push(in_wire)
                }

                self.push_control(c);
                self.modifiers.dagger = dagger;

                Ok(result)
            }
        }
    }
}

/// Delete all global phase operations in the subgraph reachable from the given entry points.
pub fn delete_phase<N: HugrNode>(
    h: &mut impl HugrMut<Node = N>,
    entry_points: impl IntoIterator<Item = N>,
) -> Result<(), ModifierResolverErrors<N>> {
    for entry_point in entry_points {
        let descendants = h.descendants(entry_point).collect::<Vec<_>>();
        for node in descendants {
            if GlobalPhase::from_optype(h.get_optype(node)).is_some() {
                h.remove_subtree(node);
                println!("found global phase: {}", node);
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::io::Write;
    use std::{fs::File, iter, path::Path, sync::Arc};

    use hugr::algorithms::{dead_code, ComposablePass};
    use hugr::builder::Container;
    use hugr::std_extensions::collections::array::ArrayOpBuilder;
    use hugr::{
        builder::{DataflowSubContainer, HugrBuilder, ModuleBuilder},
        envelope::{EnvelopeConfig, EnvelopeFormat},
        extension::{prelude::qb_t, ExtensionRegistry},
        ops::{handle::NodeHandle, CallIndirect},
        std_extensions::collections::array::array_type,
        types::{Signature, Term},
        Extension, HugrView,
    };

    use crate::extension::modifier::{CONTROL_OP_ID, DAGGER_OP_ID, MODIFIER_EXTENSION};
    use crate::{
        extension::{
            bool::BOOL_EXTENSION,
            debug::StateResult,
            rotation::{ConstRotation, ROTATION_EXTENSION},
            TKET_EXTENSION,
        },
        modifier::modifier_resolver::resolve_modifier_with_entrypoints,
    };

    trait SetUnitary {
        fn set_unitary(&mut self);
    }
    impl<T: Container> SetUnitary for T {
        fn set_unitary(&mut self) {
            self.set_metadata("unitary", 7);
        }
    }

    use super::*;

    #[test]
    fn test_global_phase() {
        let mut module = ModuleBuilder::new();
        let t_num = 0;
        let c_num = 3;
        let num = (t_num + c_num).try_into().unwrap();
        let targs = iter::repeat(qb_t()).take(t_num).collect::<Vec<_>>();
        let foo_sig = Signature::new_endo(targs);
        let call_sig = Signature::new_endo(array_type(num, qb_t()));
        let main_sig = Signature::new(vec![], array_type(num, qb_t()));

        let term_list: Vec<Term> = iter::repeat(qb_t().into()).take(t_num).collect();
        let control_op = MODIFIER_EXTENSION
            .instantiate_extension_op(
                &CONTROL_OP_ID,
                [
                    Term::BoundedNat(c_num as u64),
                    Term::new_list(term_list.clone()),
                    Term::new_list([]),
                ],
            )
            .unwrap();

        let dagger_op = MODIFIER_EXTENSION
            .instantiate_extension_op(
                &DAGGER_OP_ID,
                [Term::new_list(term_list), Term::new_list([])],
            )
            .unwrap();

        let foo = {
            let mut func = module.define_function("foo", foo_sig.clone()).unwrap();
            func.set_unitary();
            let inputs: Vec<Wire> = func.input_wires().collect();
            // let (i1,) = inputs.iter_mut().take(t_num).collect_tuple().unwrap();
            let theta = func.add_load_value(ConstRotation::new(0.5).unwrap());
            func.add_dataflow_op(GlobalPhase, vec![theta]).unwrap();
            func.finish_with_outputs(inputs).unwrap()
        };

        let _main = {
            let mut func = module.define_function("main", main_sig).unwrap();
            let mut call = func.load_func(foo.handle(), &[]).unwrap();
            call = func
                .add_dataflow_op(dagger_op, vec![call])
                .unwrap()
                .out_wire(0);
            call = func
                .add_dataflow_op(control_op, vec![call])
                .unwrap()
                .out_wire(0);

            let mut qs = Vec::new();
            for i in 0..c_num + t_num {
                qs.push({
                    let mut q = func
                        .add_dataflow_op(TketOp::QAlloc, vec![])
                        .unwrap()
                        .out_wire(0);
                    if i < c_num {
                        q = func
                            .add_dataflow_op(TketOp::H, vec![q])
                            .unwrap()
                            .out_wire(0);
                    } else {
                        // q = func.add_dataflow_op(TketOp::X, vec![q]).unwrap().out_wire(0);
                        q = func
                            .add_dataflow_op(TketOp::H, vec![q])
                            .unwrap()
                            .out_wire(0);
                    }
                    q
                });
            }

            let mut outs = func.add_new_array(qb_t(), qs).unwrap();
            let state_result = StateResult::new("input_state".to_string(), num);
            outs = func
                .add_dataflow_op(state_result, vec![outs])
                .unwrap()
                .out_wire(0);

            outs = func
                .add_dataflow_op(
                    CallIndirect {
                        signature: call_sig,
                    },
                    vec![call, outs],
                )
                .unwrap()
                .out_wire(0);

            let state_result = StateResult::new("output_state".to_string(), num);
            outs = func
                .add_dataflow_op(state_result, vec![outs])
                .unwrap()
                .out_wire(0);

            func.finish_with_outputs(vec![outs]).unwrap()
        };

        println!("Before modification:\n{}", module.hugr().mermaid_string());
        let mut h = module.finish_hugr().unwrap();
        h.validate().unwrap();
        println!("Before modification:\n{}", h.mermaid_string());

        let entrypoint = h.entrypoint().clone();
        resolve_modifier_with_entrypoints(&mut h, vec![entrypoint].into_iter()).unwrap();
        dead_code::DeadCodeElimPass::default()
            .with_entry_points(vec![_main.node()])
            .run(&mut h)
            .unwrap();
        println!("After modification\n{}", h.mermaid_string());
        {
            let f = File::create(Path::new("test_global_phase.mermaid")).unwrap();
            let mut writer = std::io::BufWriter::new(f);
            write!(writer, "{}", h.mermaid_string()).unwrap();
        }

        let env_format = EnvelopeFormat::PackageJson;
        let env_conf: EnvelopeConfig = EnvelopeConfig::new(env_format);
        let iter: Vec<Arc<Extension>> = vec![
            ROTATION_EXTENSION.to_owned(),
            TKET_EXTENSION.to_owned(),
            BOOL_EXTENSION.to_owned(),
        ];
        let regist: ExtensionRegistry = ExtensionRegistry::new(iter);
        let f = File::create(Path::new("test_global_phase.json")).unwrap();
        let writer = std::io::BufWriter::new(f);
        h.store_with_exts(writer, env_conf, &regist).unwrap();
        // println!(
        //     "hugr\n{}",
        //     h.store_str_with_exts(env_conf, &regist).unwrap()
        // );
    }
}
