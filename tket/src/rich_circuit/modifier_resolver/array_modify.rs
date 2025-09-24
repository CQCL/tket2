//! [WIP] Ad hoc implementation of array operations modification.
use hugr::{
    builder::Dataflow,
    core::HugrNode,
    extension::simple_op::MakeExtensionOp,
    hugr::hugrmut::HugrMut,
    ops::OpType,
    std_extensions::collections::{
        array::{
            ArrayKind, ArrayOp, GenericArrayOp,
            GenericArrayOpDef::{self, *},
        },
        borrow_array::{BArrayFromArray, BArrayOp, BArrayToArray},
        value_array::{VArrayFromArray, VArrayOp, VArrayToArray},
    },
};

use crate::rich_circuit::modifier_resolver::{DirWire, ModifierResolver, ModifierResolverErrors};

impl<N: HugrNode> ModifierResolver<N> {
    pub(super) fn modify_array_op(
        &mut self,
        h: &impl HugrMut<Node = N>,
        n: N,
        optype: &OpType,
        new_dfg: &mut impl Dataflow,
    ) -> Result<bool, ModifierResolverErrors<N>> {
        if let Some(op) = ArrayOp::from_optype(optype) {
            self.generic_modify_array_op(h, n, op, new_dfg)?;
        } else if let Some(op) = VArrayOp::from_optype(optype) {
            self.generic_modify_array_op(h, n, op, new_dfg)?;
        } else if let Some(op) = BArrayOp::from_optype(optype) {
            self.generic_modify_array_op(h, n, op, new_dfg)?;
        } else {
            return Ok(false);
        }

        Ok(true)
    }

    fn generic_modify_array_op<AK: ArrayKind>(
        &mut self,
        h: &impl HugrMut<Node = N>,
        n: N,
        op: GenericArrayOp<AK>,
        new_dfg: &mut impl Dataflow,
    ) -> Result<(), ModifierResolverErrors<N>> {
        let op_def = &op.def;
        if !self.modifiers().dagger {
            if *op_def != swap || self.control_num() == 0 {
                self.add_node_no_modification(new_dfg, op, h, n)?;
            } else {
                // TODO: swap needs to be converted to a quantum swap gate
                unimplemented!(
                    "Conversion from array swap to quantum swap gate is not supported yet"
                );
            }
        } else {
            let new_op_def: GenericArrayOpDef<AK> = match op_def {
                swap => {
                    // TODO: swap might need to be converted to a quantum swap gate
                    swap
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

    pub(super) fn try_array_convert(
        &mut self,
        h: &impl HugrMut<Node = N>,
        n: N,
        optype: &OpType,
        // GenericArrayConvert<AK, DIR, OtherAK>,
        new_dfg: &mut impl Dataflow,
    ) -> Result<bool, ModifierResolverErrors<N>> {
        if !self.modifiers().dagger {
            self.add_node_no_modification(new_dfg, optype.clone(), h, n)?;
            return Ok(true);
        }

        let Some(op) = optype.as_extension_op() else {
            return Ok(false);
        };

        // try some general array convert
        let node = if let Ok(op) = VArrayToArray::from_extension_op(op) {
            new_dfg.add_child_node(VArrayFromArray::new(op.elem_ty, op.size))
        } else if let Ok(op) = VArrayFromArray::from_extension_op(op) {
            new_dfg.add_child_node(VArrayToArray::new(op.elem_ty, op.size))
        } else if let Ok(op) = BArrayToArray::from_extension_op(op) {
            new_dfg.add_child_node(BArrayFromArray::new(op.elem_ty, op.size))
        } else if let Ok(op) = BArrayFromArray::from_extension_op(op) {
            new_dfg.add_child_node(BArrayToArray::new(op.elem_ty, op.size))
        } else {
            return Ok(false);
        };

        for port in h.all_node_ports(n) {
            let wire = DirWire::new(node, port).reverse();
            self.map_insert(DirWire(n, port), wire)?;
        }

        Ok(true)
    }
}
