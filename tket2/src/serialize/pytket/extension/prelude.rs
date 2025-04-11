//! Encoder and decoder for tket2 operations with native pytket counterparts.

use super::Tk1Encoder;
use crate::serialize::pytket::encoder::{RegisterCount, Tk1EncoderContext, TrackedValue};
use crate::serialize::pytket::Tk1ConvertError;
use crate::Circuit;
use hugr::extension::prelude::{TupleOpDef, PRELUDE_ID};
use hugr::extension::simple_op::MakeExtensionOp;
use hugr::extension::ExtensionId;
use hugr::ops::{DataflowOpTrait, ExtensionOp};
use hugr::types::TypeArg;
use hugr::HugrView;

/// Encoder for [prelude](hugr::extension::prelude) operations.
#[derive(Debug, Clone, Default)]
pub struct PreludeEncoder;

impl<H: HugrView> Tk1Encoder<H> for PreludeEncoder {
    fn extensions(&self) -> Option<Vec<ExtensionId>> {
        Some(vec![PRELUDE_ID])
    }

    fn op_to_pytket(
        &self,
        node: H::Node,
        op: &ExtensionOp,
        circ: &Circuit<H>,
        encoder: &mut Tk1EncoderContext<H>,
    ) -> Result<bool, Tk1ConvertError<H::Node>> {
        if let Ok(tuple_op) = TupleOpDef::from_extension_op(op) {
            return self.tuple_op_to_pytket(node, op, &tuple_op, circ, encoder);
        };

        return Ok(false);
    }

    fn type_to_pytket(
        &self,
        typ: &hugr::types::CustomType,
    ) -> Result<Option<RegisterCount>, Tk1ConvertError<<H>::Node>> {
        match typ.name().as_str() {
            "usize" => Ok(Some(RegisterCount::only_bits(64))),
            "qubit" => Ok(Some(RegisterCount::only_qubits(1))),
            _ => Ok(None),
        }
    }
}

impl PreludeEncoder {
    /// Encode a prelude tuple operation.
    ///
    /// These just bundle/unbundle the values of the inputs/outputs. Since
    /// pytket types are already flattened, the translation of these is just a
    /// no-op.
    fn tuple_op_to_pytket<H: HugrView>(
        &self,
        node: H::Node,
        op: &ExtensionOp,
        tuple_op: &TupleOpDef,
        circ: &Circuit<H>,
        encoder: &mut Tk1EncoderContext<H>,
    ) -> Result<bool, Tk1ConvertError<H::Node>> {
        if !matches!(tuple_op, TupleOpDef::MakeTuple | TupleOpDef::UnpackTuple) {
            // Unknown operation
            return Ok(false);
        };

        // First, check if we are working with supported types.
        //
        // If any of the types cannot be translated to a pytket type, we return
        // false so the operation is marked as unsupported as a whole.
        let args = op.args().first();
        match args {
            Some(TypeArg::Sequence { elems }) => {
                for arg in elems {
                    let TypeArg::Type { ty } = arg else {
                        return Ok(false);
                    };
                    let count = encoder.config().type_to_pytket(ty)?;
                    if count.is_none() {
                        return Ok(false);
                    }
                }
            }
            _ => return Ok(false),
        };

        // Now we can gather all inputs and assign them to the node outputs transparently.
        let input_values = encoder.get_input_values(node, circ)?;
        let mut qubits = input_values.qubits.into_iter();
        let mut bits = input_values.bits.into_iter();
        let mut params = input_values.params.into_iter();

        let signature = op.signature();
        for (out_port, ty) in circ.hugr().node_outputs(node).zip(signature.output.iter()) {
            let wire = hugr::Wire::new(node, out_port);
            let Some(count) = encoder.config().type_to_pytket(ty)? else {
                return Err(Tk1ConvertError::custom(
                    "Found an unsupported type while encoding a TupleOp.",
                ));
            };
            let mut values: Vec<TrackedValue> = Vec::with_capacity(count.total());
            for _ in 0..count.qubits {
                let Some(qb) = qubits.next() else {
                    return Err(Tk1ConvertError::custom(
                        "Not enough qubits in the input values for a TupleOp.",
                    ));
                };
                values.push(qb.into());
            }
            for _ in 0..count.bits {
                let Some(bit) = bits.next() else {
                    return Err(Tk1ConvertError::custom(
                        "Not enough bits in the input values for a TupleOp.",
                    ));
                };
                values.push(bit.into());
            }
            for _ in 0..count.params {
                let Some(param) = params.next() else {
                    return Err(Tk1ConvertError::custom(
                        "Not enough parameters in the input values for a TupleOp.",
                    ));
                };
                values.push(param.into());
            }
            encoder.values.register_values(wire, values, circ)?;
        }

        Ok(true)
    }
}
