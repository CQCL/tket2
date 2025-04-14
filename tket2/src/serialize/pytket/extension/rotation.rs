//! Encoder and decoder for rotation operations.

use super::Tk1Encoder;
use crate::extension::rotation::{
    ConstRotation, RotationOp, ROTATION_EXTENSION_ID, ROTATION_TYPE_ID,
};
use crate::serialize::pytket::encoder::{RegisterCount, Tk1EncoderContext, TrackedValues};
use crate::serialize::pytket::Tk1ConvertError;
use crate::Circuit;
use hugr::extension::simple_op::MakeExtensionOp;
use hugr::extension::ExtensionId;
use hugr::ops::constant::OpaqueValue;
use hugr::ops::ExtensionOp;
use hugr::HugrView;

/// Encoder for [prelude](hugr::extension::prelude) operations.
#[derive(Debug, Clone, Default)]
pub struct RotationEncoder;

impl<H: HugrView> Tk1Encoder<H> for RotationEncoder {
    fn extensions(&self) -> Option<Vec<ExtensionId>> {
        Some(vec![ROTATION_EXTENSION_ID])
    }

    fn op_to_pytket(
        &self,
        node: H::Node,
        op: &ExtensionOp,
        circ: &Circuit<H>,
        encoder: &mut Tk1EncoderContext<H>,
    ) -> Result<bool, Tk1ConvertError<H::Node>> {
        let Ok(rot_op) = RotationOp::from_extension_op(op) else {
            return Ok(false);
        };

        match rot_op {
            RotationOp::from_halfturns_unchecked | RotationOp::to_halfturns => {
                encoder.emit_transparent_node(node, circ, |_, ps| ps.first().cloned())?;
                Ok(true)
            }
            RotationOp::from_halfturns => {
                // Unsupported due to having an Option as output.
                Ok(false)
            }
            _ => {
                encoder.emit_transparent_node(node, circ, |_, ps| {
                    RotationEncoder::encode_rotation_op(&rot_op, ps)
                })?;
                Ok(true)
            }
        }
    }

    fn type_to_pytket(
        &self,
        typ: &hugr::types::CustomType,
    ) -> Result<Option<RegisterCount>, Tk1ConvertError<<H>::Node>> {
        match typ.name() == &ROTATION_TYPE_ID {
            true => Ok(Some(RegisterCount::only_params(1))),
            false => Ok(None),
        }
    }

    fn const_to_pytket(
        &self,
        value: &OpaqueValue,
        encoder: &mut Tk1EncoderContext<H>,
    ) -> Result<Option<TrackedValues>, Tk1ConvertError<H::Node>> {
        let Some(const_f) = value.value().downcast_ref::<ConstRotation>() else {
            return Ok(None);
        };

        let param = encoder.values.new_param(const_f.half_turns());

        let mut values = TrackedValues::default();
        values.params.push(param);
        Ok(Some(values))
    }
}

impl RotationEncoder {
    /// Encode a rotation operation into a pytket param expression.
    fn encode_rotation_op(op: &RotationOp, inputs: &[String]) -> Option<String> {
        let s = match op {
            RotationOp::radd => format!("({} + {})", inputs[0], inputs[1]),
            RotationOp::to_halfturns
            | RotationOp::from_halfturns_unchecked
            | RotationOp::from_halfturns => return None,
        };
        Some(s)
    }
}
