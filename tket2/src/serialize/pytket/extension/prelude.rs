//! Encoder and decoder for tket2 operations with native pytket counterparts.

use super::PytketEmitter;
use crate::serialize::pytket::encoder::{EncodeStatus, RegisterCount, Tk1EncoderContext};
use crate::serialize::pytket::Tk1ConvertError;
use crate::Circuit;
use hugr::extension::prelude::{BarrierDef, TupleOpDef, PRELUDE_ID};
use hugr::extension::simple_op::MakeExtensionOp;
use hugr::extension::ExtensionId;
use hugr::ops::ExtensionOp;
use hugr::types::TypeArg;
use hugr::HugrView;
use tket_json_rs::optype::OpType as Tk1OpType;

/// Encoder for [prelude](hugr::extension::prelude) operations.
#[derive(Debug, Clone, Default)]
pub struct PreludeEmitter;

impl<H: HugrView> PytketEmitter<H> for PreludeEmitter {
    fn extensions(&self) -> Option<Vec<ExtensionId>> {
        Some(vec![PRELUDE_ID])
    }

    fn op_to_pytket(
        &self,
        node: H::Node,
        op: &ExtensionOp,
        circ: &Circuit<H>,
        encoder: &mut Tk1EncoderContext<H>,
    ) -> Result<EncodeStatus, Tk1ConvertError<H::Node>> {
        if let Ok(tuple_op) = TupleOpDef::from_extension_op(op) {
            return self.tuple_op_to_pytket(node, op, &tuple_op, circ, encoder);
        };
        if let Ok(_barrier) = BarrierDef::from_extension_op(op) {
            encoder.emit_node(Tk1OpType::Barrier, node, circ)?;
            return Ok(EncodeStatus::Success);
        };
        Ok(EncodeStatus::Unsupported)
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

impl PreludeEmitter {
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
    ) -> Result<EncodeStatus, Tk1ConvertError<H::Node>> {
        if !matches!(tuple_op, TupleOpDef::MakeTuple | TupleOpDef::UnpackTuple) {
            // Unknown operation
            return Ok(EncodeStatus::Unsupported);
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
                        return Ok(EncodeStatus::Unsupported);
                    };
                    let count = encoder.config().type_to_pytket(ty)?;
                    if count.is_none() {
                        return Ok(EncodeStatus::Unsupported);
                    }
                }
            }
            _ => return Ok(EncodeStatus::Unsupported),
        };

        // Now we can gather all inputs and assign them to the node outputs transparently.
        encoder.emit_transparent_node(node, circ, |ps| ps.input_params.to_owned())?;

        Ok(EncodeStatus::Success)
    }
}
