//! Encoder and decoder for the tket2.bool extension

use super::PytketEmitter;
use crate::extension::bool::{BoolOp, ConstBool, BOOL_EXTENSION_ID, BOOL_TYPE_NAME};
use crate::serialize::pytket::encoder::{
    make_tk1_classical_expression, make_tk1_classical_operation, EncodeStatus, RegisterCount,
    Tk1EncoderContext, TrackedValues,
};
use crate::serialize::pytket::Tk1ConvertError;
use crate::Circuit;
use hugr::extension::simple_op::MakeExtensionOp;
use hugr::extension::ExtensionId;
use hugr::ops::constant::OpaqueValue;
use hugr::ops::ExtensionOp;
use hugr::HugrView;
use itertools::Itertools;
use tket_json_rs::clexpr::op::ClOp;
use tket_json_rs::clexpr::operator::{ClArgument, ClOperator, ClTerminal, ClVariable};

/// Encoder for [prelude](hugr::extension::prelude) operations.
#[derive(Debug, Clone, Default)]
pub struct BoolEmitter;

impl<H: HugrView> PytketEmitter<H> for BoolEmitter {
    fn extensions(&self) -> Option<Vec<ExtensionId>> {
        Some(vec![BOOL_EXTENSION_ID])
    }

    fn op_to_pytket(
        &self,
        node: H::Node,
        op: &ExtensionOp,
        circ: &Circuit<H>,
        encoder: &mut Tk1EncoderContext<H>,
    ) -> Result<EncodeStatus, Tk1ConvertError<H::Node>> {
        let Ok(rot_op) = BoolOp::from_extension_op(op) else {
            return Ok(EncodeStatus::Unsupported);
        };

        let (num_inputs, num_outputs, clop) = match rot_op {
            // Conversion ops between native bools and `tket2.bool`.
            // Both are represented as a pytket bit, so this is a no-op.
            BoolOp::read | BoolOp::make_opaque => {
                encoder.emit_transparent_node(node, circ, |_| Vec::new())?;
                return Ok(EncodeStatus::Success);
            }
            BoolOp::eq => (2, 1, ClOp::BitEq),
            BoolOp::not => (1, 1, ClOp::BitNot),
            BoolOp::and => (2, 1, ClOp::BitAnd),
            BoolOp::or => (2, 1, ClOp::BitOr),
            BoolOp::xor => (2, 1, ClOp::BitXor),
        };

        // We assume here all operations are a single expression node, with only
        // variable inputs. If new [`BoolOp`]s are added that do not follow
        // this, the following code will need to be adjusted.
        let bit_count = (num_inputs + num_outputs) as usize;
        let output_bits = (0..num_outputs).collect_vec();
        let mut expression = ClOperator::default();
        expression.op = clop;
        expression.args = (0..num_inputs)
            .map(|i| ClArgument::Terminal(ClTerminal::Variable(ClVariable::Bit { index: i })))
            .collect_vec();

        let op = make_tk1_classical_expression(bit_count, &output_bits, &[], expression);
        encoder.emit_node_command(node, circ, |_args| Vec::new(), move |_| op)?;
        Ok(EncodeStatus::Success)
    }

    fn type_to_pytket(
        &self,
        typ: &hugr::types::CustomType,
    ) -> Result<Option<RegisterCount>, Tk1ConvertError<<H>::Node>> {
        if typ.name() == &*BOOL_TYPE_NAME {
            Ok(Some(RegisterCount::only_bits(1)))
        } else {
            Ok(None)
        }
    }

    fn const_to_pytket(
        &self,
        value: &OpaqueValue,
        encoder: &mut Tk1EncoderContext<H>,
    ) -> Result<Option<TrackedValues>, Tk1ConvertError<H::Node>> {
        let Some(const_b) = value.value().downcast_ref::<ConstBool>() else {
            return Ok(None);
        };

        let new_bit = encoder.values.new_bit();
        if const_b.value() {
            let op = set_bits_op(&[true]);
            encoder.emit_command(op, &[], &[new_bit], None);
        }

        Ok(Some(TrackedValues::new_bits([new_bit])))
    }
}

/// Return a pytket operation setting the values of a list of bits.
pub(crate) fn set_bits_op(values: &[bool]) -> tket_json_rs::circuit_json::Operation {
    make_tk1_classical_operation(
        tket_json_rs::OpType::SetBits,
        values.len(),
        tket_json_rs::circuit_json::Classical::SetBits {
            values: values.to_vec(),
        },
    )
}
