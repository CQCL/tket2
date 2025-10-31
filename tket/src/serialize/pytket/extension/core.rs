//! Decoder for native HUGR structures generated from pytket operations.
//!
//! These core structures are handled natively by the pytket encoder, so we
//! don't need to implement a
//! [`PytketEmitter`][crate::serialize::pytket::extension::PytketEmitter] for
//! them.

use crate::extension::rotation::rotation_type;
use crate::serialize::pytket::decoder::{
    DecodeStatus, LoadedParameter, PytketDecoderContext, TrackedBit, TrackedQubit,
};
use crate::serialize::pytket::extension::PytketDecoder;
use crate::serialize::pytket::opaque::{OpaqueSubgraphPayload, OPGROUP_OPAQUE_HUGR};
use crate::serialize::pytket::{DecodeInsertionTarget, DecodeOptions, PytketDecodeError};
use hugr::builder::Container;
use hugr::extension::prelude::{bool_t, qb_t};
use hugr::ops::handle::NodeHandle;
use hugr::types::{Signature, Type};
use itertools::Itertools;
use tket_json_rs::circuit_json::Operation as PytketOperation;
use tket_json_rs::opbox::OpBox;
use tket_json_rs::optype::OpType as PytketOptype;

/// Decoder for native HUGR structures.
#[derive(Debug, Clone, Default)]
pub struct CoreDecoder;

impl PytketDecoder for CoreDecoder {
    fn op_types(&self) -> Vec<PytketOptype> {
        vec![PytketOptype::Barrier, PytketOptype::CircBox]
    }

    fn op_to_hugr<'h>(
        &self,
        op: &PytketOperation,
        qubits: &[TrackedQubit],
        bits: &[TrackedBit],
        params: &[LoadedParameter],
        opgroup: Option<&str>,
        decoder: &mut PytketDecoderContext<'h>,
    ) -> Result<DecodeStatus, PytketDecodeError> {
        match &op {
            PytketOperation {
                op_type: PytketOptype::Barrier,
                data: Some(payload),
                ..
            } if opgroup == Some(OPGROUP_OPAQUE_HUGR) => {
                let Ok(payload) = OpaqueSubgraphPayload::load_str(
                    payload,
                    decoder.options().extension_registry(),
                ) else {
                    // Payload is invalid. We don't error here to avoid
                    // panicking on corrupted/old user submissions.
                    return Ok(DecodeStatus::Unsupported);
                };
                decoder.insert_subgraph_from_payload(qubits, bits, params, &payload)
            }
            PytketOperation {
                op_type: PytketOptype::CircBox,
                op_box:
                    Some(OpBox::CircBox {
                        id: _id,
                        circuit: serial_circuit,
                    }),
                ..
            } => {
                // We have no way to distinguish between input and output bits
                // in the circuit box, so we assume all bits are both inputs and
                // outputs here.
                let circ_inputs: Vec<Type> = itertools::repeat_n(qb_t(), qubits.len())
                    .chain(itertools::repeat_n(bool_t(), bits.len()))
                    .chain(itertools::repeat_n(rotation_type(), params.len()))
                    .collect_vec();
                let circ_outputs: Vec<Type> = itertools::repeat_n(qb_t(), qubits.len())
                    .chain(itertools::repeat_n(bool_t(), bits.len()))
                    .collect_vec();
                let circ_signature = Signature::new(circ_inputs, circ_outputs);

                // Decode the boxed circuit into a DFG node in the Hugr.
                let options = DecodeOptions::new()
                    .with_config(decoder.config().clone())
                    .with_signature(circ_signature);
                let target = DecodeInsertionTarget::Region {
                    parent: decoder.builder.container_node(),
                };

                // Decode the circuit box into a DFG node in the region.
                let mut nested_decoder = PytketDecoderContext::new(
                    serial_circuit,
                    decoder.builder.hugr_mut(),
                    target,
                    options,
                )?;
                if let Some(opaque_subgraphs) = decoder.opaque_subgraphs {
                    nested_decoder.register_opaque_subgraphs(opaque_subgraphs);
                }
                nested_decoder.run_decoder(&serial_circuit.commands)?;
                let internal = nested_decoder.finish()?.node();

                decoder
                    .wire_up_node(internal, qubits, qubits, bits, bits, params)
                    .map_err(|e| e.hugr_op("DFG"))?;

                Ok(DecodeStatus::Success)
            }
            _ => Ok(DecodeStatus::Unsupported),
        }
    }
}
