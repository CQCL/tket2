//! Encoder and decoder for floating point operations.

use super::PytketEmitter;
use crate::serialize::pytket::encoder::{RegisterCount, Tk1EncoderContext, TrackedValues};
use crate::serialize::pytket::Tk1ConvertError;
use crate::Circuit;
use hugr::extension::simple_op::MakeExtensionOp;
use hugr::extension::ExtensionId;
use hugr::ops::constant::OpaqueValue;
use hugr::ops::ExtensionOp;
use hugr::std_extensions::arithmetic::float_ops::FloatOps;
use hugr::std_extensions::arithmetic::{float_ops, float_types};
use hugr::HugrView;

/// Encoder for [prelude](hugr::extension::prelude) operations.
#[derive(Debug, Clone, Default)]
pub struct FloatEmitter;

impl<H: HugrView> PytketEmitter<H> for FloatEmitter {
    fn extensions(&self) -> Option<Vec<ExtensionId>> {
        Some(vec![float_ops::EXTENSION_ID, float_types::EXTENSION_ID])
    }

    fn op_to_pytket(
        &self,
        node: H::Node,
        op: &ExtensionOp,
        circ: &Circuit<H>,
        encoder: &mut Tk1EncoderContext<H>,
    ) -> Result<bool, Tk1ConvertError<H::Node>> {
        let Ok(rot_op) = FloatOps::from_extension_op(op) else {
            return Ok(false);
        };

        match rot_op {
            FloatOps::fadd
            | FloatOps::fsub
            | FloatOps::fneg
            | FloatOps::fmul
            | FloatOps::fdiv
            | FloatOps::fpow
            | FloatOps::ffloor
            | FloatOps::fceil
            | FloatOps::fround
            | FloatOps::fmax
            | FloatOps::fmin
            | FloatOps::fabs => {
                encoder.emit_transparent_node(node, circ, |ps| {
                    match FloatEmitter::encode_rotation_op(&rot_op, ps.input_params) {
                        Some(s) => vec![s],
                        None => Vec::new(),
                    }
                })?;
                Ok(true)
            }
            _ => Ok(false),
        }
    }

    fn type_to_pytket(
        &self,
        typ: &hugr::types::CustomType,
    ) -> Result<Option<RegisterCount>, Tk1ConvertError<<H>::Node>> {
        match typ.name() == &float_types::FLOAT_TYPE_ID {
            true => Ok(Some(RegisterCount::only_params(1))),
            false => Ok(None),
        }
    }

    fn const_to_pytket(
        &self,
        value: &OpaqueValue,
        encoder: &mut Tk1EncoderContext<H>,
    ) -> Result<Option<TrackedValues>, Tk1ConvertError<H::Node>> {
        let Some(const_f) = value.value().downcast_ref::<float_types::ConstF64>() else {
            return Ok(None);
        };

        let float = const_f.value();
        // Special case for pi rotations
        let val = if float == std::f64::consts::PI {
            "pi".to_string()
        } else {
            float.to_string()
        };

        let param = encoder.values.new_param(val);

        let mut values = TrackedValues::default();
        values.params.push(param);
        Ok(Some(values))
    }
}

impl FloatEmitter {
    /// Encode a rotation operation into a pytket param expression.
    fn encode_rotation_op(op: &FloatOps, inputs: &[String]) -> Option<String> {
        let s = match op {
            FloatOps::fadd => format!("({} + {})", inputs[0], inputs[1]),
            FloatOps::fsub => format!("({} - {})", inputs[0], inputs[1]),
            FloatOps::fneg => format!("(-{})", inputs[0]),
            FloatOps::fmul => format!("({} * {})", inputs[0], inputs[1]),
            FloatOps::fdiv => format!("({} / {})", inputs[0], inputs[1]),
            FloatOps::fpow => format!("({} ** {})", inputs[0], inputs[1]),
            FloatOps::ffloor => format!("floor({})", inputs[0]),
            FloatOps::fceil => format!("ceil({})", inputs[0]),
            FloatOps::fround => format!("round({})", inputs[0]),
            FloatOps::fmax => format!("max({}, {})", inputs[0], inputs[1]),
            FloatOps::fmin => format!("min({}, {})", inputs[0], inputs[1]),
            FloatOps::fabs => format!("abs({})", inputs[0]),
            _ => return None,
        };
        Some(s)
    }
}
