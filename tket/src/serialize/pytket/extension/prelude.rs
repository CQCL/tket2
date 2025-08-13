//! Encoder and decoder for tket operations with native pytket counterparts.

use super::PytketEmitter;
use crate::serialize::pytket::config::TypeTranslatorSet;
use crate::serialize::pytket::decoder::{DecodeStatus, PytketDecoderContext, TrackedWires};
use crate::serialize::pytket::encoder::{EncodeStatus, PytketEncoderContext};
use crate::serialize::pytket::extension::{PytketDecoder, PytketTypeTranslator, RegisterCount};
use crate::serialize::pytket::{PytketDecodeError, PytketEncodeError};
use crate::Circuit;
use hugr::builder::Dataflow;
use hugr::extension::prelude::{qb_t, BarrierDef, Noop, TupleOpDef, PRELUDE_ID};
use hugr::extension::simple_op::MakeExtensionOp;
use hugr::extension::ExtensionId;
use hugr::ops::handle::NodeHandle;
use hugr::ops::{ExtensionOp, OpType};
use hugr::types::TypeArg;
use hugr::HugrView;
use itertools::Itertools as _;
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
        encoder: &mut PytketEncoderContext<H>,
    ) -> Result<EncodeStatus, PytketEncodeError<H::Node>> {
        if let Ok(tuple_op) = TupleOpDef::from_extension_op(op) {
            return self.tuple_op_to_pytket(node, op, &tuple_op, circ, encoder);
        };
        if let Ok(_barrier) = BarrierDef::from_extension_op(op) {
            encoder.emit_node(Tk1OpType::Barrier, node, circ)?;
            return Ok(EncodeStatus::Success);
        };
        Ok(EncodeStatus::Unsupported)
    }
}

impl PytketTypeTranslator for PreludeEmitter {
    fn extensions(&self) -> Vec<ExtensionId> {
        vec![PRELUDE_ID]
    }

    fn type_to_pytket(
        &self,
        typ: &hugr::types::CustomType,
        _set: &TypeTranslatorSet,
    ) -> Option<RegisterCount> {
        match typ.name().as_str() {
            "usize" => Some(RegisterCount::only_bits(64)),
            "qubit" => Some(RegisterCount::only_qubits(1)),
            _ => None,
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
        encoder: &mut PytketEncoderContext<H>,
    ) -> Result<EncodeStatus, PytketEncodeError<H::Node>> {
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
            Some(TypeArg::Tuple(elems)) | Some(TypeArg::List(elems)) => {
                for arg in elems {
                    let TypeArg::Runtime(ty) = arg else {
                        return Ok(EncodeStatus::Unsupported);
                    };
                    let count = encoder.config().type_to_pytket(ty);
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

impl PytketDecoder for PreludeEmitter {
    fn op_types(&self) -> Vec<Tk1OpType> {
        vec![Tk1OpType::noop, Tk1OpType::Measure]
    }

    fn op_to_hugr<'h>(
        &self,
        op: &tket_json_rs::circuit_json::Operation,
        wires: &TrackedWires,
        opgroup: Option<&str>,
        decoder: &mut PytketDecoderContext<'h>,
    ) -> Result<DecodeStatus, PytketDecodeError> {
        // Qubits, bits and parameters that will be used to register the node outputs.
        //
        // These should be modified by the match branches if the node does not have all
        // its input registers in the outputs.
        let qubits = wires.qubits(decoder).collect_vec();
        let bits = wires.bits(decoder).collect_vec();

        let op: OpType = match op.op_type {
            Tk1OpType::noop => Noop::new(qb_t()).into(),
            Tk1OpType::Barrier => {
                // We use pytket barriers in the pytket encoder framework to store
                // HUGRs that cannot be represented in pytket.
                //
                // We take care here and detect when that happens.
                // TODO: For now, we just say the conversion is unsupported instead of extracting the Hugr.
                if opgroup == Some("UNSUPPORTED_HUGR") {
                    return Ok(DecodeStatus::Unsupported);
                }

                let types = wires.wire_types().cloned().collect_vec();
                hugr::extension::prelude::Barrier::new(types).into()
            }
            _ => return Ok(DecodeStatus::Unsupported),
        };

        // Convert parameter inputs to rotation types
        let param_wires = wires
            .iter_parameters()
            .map(|p| p.as_rotation(&mut decoder.builder).wire)
            .collect_vec();
        let value_wires = wires.value_wires();

        let node = decoder
            .builder
            .add_dataflow_op(op, value_wires.chain(param_wires.into_iter()))
            .map_err(PytketDecodeError::custom)?;

        decoder.register_node_outputs(node.node(), qubits, bits)?;

        Ok(DecodeStatus::Success)
    }
}
