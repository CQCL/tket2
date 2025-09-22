//! [WIP] Ad hoc implementation of array operations modification.
use hugr::{
    builder::Dataflow,
    core::HugrNode,
    extension::simple_op::MakeExtensionOp,
    hugr::hugrmut::HugrMut,
    ops::OpType,
    std_extensions::collections::array::{
        Array, ArrayKind, ArrayOp, Direction, GenericArrayConvert, GenericArrayOp,
        GenericArrayOpDef::*,
    },
};

use crate::rich_circuit::modifier_resolver::{DirWire, ModifierResolver, ModifierResolverErrors};

impl<N: HugrNode> ModifierResolver<N> {
    pub(super) fn modify_array_op(
        &mut self,
        h: &impl HugrMut<Node = N>,
        n: N,
        op: GenericArrayOp<Array>,
        new_dfg: &mut impl Dataflow,
    ) -> Result<(), ModifierResolverErrors<N>> {
        let op_def = &op.def;
        if !self.modifiers().dagger {
            if *op_def != swap {
                self.add_node_no_modification(new_dfg, op, h, n)?;
            } else {
                // TODO: swap might need to be converted to a quantum swap gate
                todo!()
            }
        } else {
            let new_op_def = match op_def {
                swap => {
                    // TODO: swap might need to be converted to a quantum swap gate
                    op_def.clone()
                }
                new_array => unpack,
                unpack => new_array,
                get => todo!(),
                set => todo!(),
                pop_left => todo!(),
                pop_right => todo!(),
                discard_empty => todo!(),
                _phantom(phantom_data, never) => todo!(),
                _ => todo!(),
            };
            let new_op = new_op_def.to_concrete(op.elem_ty, op.size);
            let node = new_dfg.add_child_node(new_op);
            for port in h.all_node_ports(n) {
                let wire = DirWire::new(node, port).reverse();
                self.map_insert(DirWire(n, port), wire)?;
            }
        }
        Ok(())
    }

    pub(super) fn modify_value_array_op<AK, const DIR: Direction, OtherAK>(
        &mut self,
        h: &impl HugrMut<Node = N>,
        n: N,
        op: GenericArrayConvert<AK, DIR, OtherAK>,
        new_dfg: &mut impl Dataflow,
    ) -> Result<(), ModifierResolverErrors<N>>
    where
        AK: ArrayKind,
        OtherAK: ArrayKind,
    {
        let REV = !DIR;
        // let new_op = GenericArrayConvert::<AK, !REV, OtherAK>::new(op.elem_ty, op.size);
        Ok(())
        // let op_def = &op.def;
        // if !self.modifiers().dagger {
        //     if *op_def != swap {
        //         self.add_node_no_modification(new_dfg, op, h, n)?;

        // TODO
        // TODO
        // TODO
        // TODO
        // TODO
        // TODO
        // TODO
        // TODO
        // TODO
        // TODO
        // TODO
        // TODO
        // TODO
        // TODO
        // TODO

        //     } else {
        //     }
        // }
    }
}
