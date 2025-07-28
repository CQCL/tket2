//! Encoder and decoder for rotation operations.

use super::PytketEmitter;
use crate::extension::rotation::{
    ConstRotation, RotationOp, ROTATION_EXTENSION_ID, ROTATION_TYPE_ID,
};
use crate::serialize::pytket::encoder::{
    EncodeStatus, RegisterCount, Tk1EncoderContext, TrackedValues,
};
use crate::serialize::pytket::Tk1ConvertError;
use crate::Circuit;
use hugr::extension::simple_op::MakeExtensionOp;
use hugr::extension::ExtensionId;
use hugr::ops::constant::OpaqueValue;
use hugr::ops::ExtensionOp;
use hugr::HugrView;
use itertools::Itertools;

/// Encoder for [prelude](hugr::extension::prelude) operations.
#[derive(Debug, Clone, Default)]
pub struct RotationEmitter;

impl<H: HugrView> PytketEmitter<H> for RotationEmitter {
    fn extensions(&self) -> Option<Vec<ExtensionId>> {
        Some(vec![ROTATION_EXTENSION_ID])
    }

    fn op_to_pytket(
        &self,
        node: H::Node,
        op: &ExtensionOp,
        circ: &Circuit<H>,
        encoder: &mut Tk1EncoderContext<H>,
    ) -> Result<EncodeStatus, Tk1ConvertError<H::Node>> {
        let Ok(rot_op) = RotationOp::from_extension_op(op) else {
            return Ok(EncodeStatus::Unsupported);
        };

        match rot_op {
            RotationOp::from_halfturns_unchecked | RotationOp::to_halfturns => {
                encoder.emit_transparent_node(node, circ, |ps| vec![ps.input_params[0].clone()])?;
                Ok(EncodeStatus::Success)
            }
            RotationOp::from_halfturns => {
                // Unsupported due to having an Option as output.
                Ok(EncodeStatus::Unsupported)
            }
            _ => {
                encoder.emit_transparent_node(node, circ, |ps| {
                    RotationEmitter::encode_rotation_op(&rot_op, ps.input_params)
                        .into_iter()
                        .collect_vec()
                })?;
                Ok(EncodeStatus::Success)
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
        Ok(Some(TrackedValues::new_params([param])))
    }
}

impl RotationEmitter {
    /// Encode a rotation operation into a pytket param expression.
    fn encode_rotation_op(op: &RotationOp, inputs: &[String]) -> Option<String> {
        let s = match op {
            RotationOp::radd => format!("({}) + ({})", inputs[0], inputs[1]),
            RotationOp::to_halfturns
            | RotationOp::from_halfturns_unchecked
            | RotationOp::from_halfturns => return None,
        };
        Some(s)
    }
}
