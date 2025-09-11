//! Decoder for native HUGR structures generated from pytket operations.
//!
//! These do not have a corresponding encoder since they are not represented as
//! `ExtensionOp`s nor `ExtensionType`s in the HUGR.

use crate::extension::rotation::rotation_type;
use crate::serialize::pytket::decoder::{
    DecodeStatus, LoadedParameter, PytketDecoderContext, TrackedBit, TrackedQubit,
};
use crate::serialize::pytket::extension::PytketDecoder;
use crate::serialize::pytket::{DecodeInsertionTarget, DecodeOptions, PytketDecodeError};
use crate::serialize::TKETDecode;
use hugr::builder::Container;
use hugr::extension::prelude::{bool_t, qb_t};
use hugr::types::{Signature, Type};
use itertools::Itertools;
use tket_json_rs::opbox::OpBox;
use tket_json_rs::optype::OpType as PytketOptype;

/// Decoder for native HUGR structures.
#[derive(Debug, Clone, Default)]
pub struct CoreDecoder;

impl PytketDecoder for CoreDecoder {
    fn op_types(&self) -> Vec<PytketOptype> {
        vec![PytketOptype::CircBox]
    }

    fn op_to_hugr<'h>(
        &self,
        op: &tket_json_rs::circuit_json::Operation,
        qubits: &[TrackedQubit],
        bits: &[TrackedBit],
        params: &[LoadedParameter],
        _opgroup: Option<&str>,
        decoder: &mut PytketDecoderContext<'h>,
    ) -> Result<DecodeStatus, PytketDecodeError> {
        match op.op_type {
            PytketOptype::CircBox => {
                let Some(OpBox::CircBox { id: _id, circuit }) = &op.op_box else {
                    return Ok(DecodeStatus::Unsupported);
                };

                // We have no way to distinguish between input and output bits in the circuit box, so we assume all bits are outputs here.
                //
                // TODO: Pass the registers both as inputs and outputs once this is implemented
                // <https://github.com/CQCL/tket2/issues/1124>
                let circ_inputs: Vec<Type> = itertools::repeat_n(qb_t(), qubits.len())
                    .chain(itertools::repeat_n(rotation_type(), params.len()))
                    .collect_vec();
                let circ_outputs: Vec<Type> = itertools::repeat_n(qb_t(), qubits.len())
                    .chain(itertools::repeat_n(bool_t(), bits.len()))
                    .collect_vec();
                let circ_signature = Signature::new(circ_inputs, circ_outputs);

                // Decode the boxed circuit into a new Hugr
                let config = decoder.config().clone();
                let insertion_target = decoder.builder.container_node();
                let internal = circuit.decode_inplace(
                    decoder.builder.hugr_mut(),
                    DecodeInsertionTarget::Region {
                        parent: insertion_target,
                    },
                    DecodeOptions::new()
                        .with_config(config)
                        .with_signature(circ_signature),
                )?;

                // Create a DFG node in the parent Hugr that will contain the decoded circuit.
                decoder
                    .wire_up_node(internal, qubits, bits, params)
                    .map_err(|e| e.hugr_op("DFG"))?;

                Ok(DecodeStatus::Success)
            }
            _ => Ok(DecodeStatus::Unsupported),
        }
    }
}
