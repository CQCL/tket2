//! Decoder for native HUGR structures generated from pytket operations.
//!
//! These core structures are handled natively by the pytket encoder, so we
//! don't need to implement a
//! [`PytketEmitter`][crate::serialize::pytket::extension::PytketEmitter] for
//! them.

use std::collections::HashMap;
use std::ops::RangeTo;
use std::sync::Arc;

use crate::extension::rotation::rotation_type;
use crate::serialize::pytket::decoder::{
    DecodeStatus, FoundWire, LoadedParameter, PytketDecoderContext, TrackedBit, TrackedQubit,
};
use crate::serialize::pytket::extension::{PytketDecoder, RegisterCount};
use crate::serialize::pytket::opaque::{EncodedEdgeID, OpaqueSubgraphPayload, OPGROUP_OPAQUE_HUGR};
use crate::serialize::pytket::{
    DecodeInsertionTarget, DecodeOptions, PytketDecodeError, PytketDecodeErrorInner,
    PytketDecoderConfig,
};
use crate::serialize::TKETDecode;
use hugr::builder::Container;
use hugr::extension::prelude::{bool_t, qb_t};
use hugr::hugr::hugrmut::HugrMut;
use hugr::types::{Signature, Type};
use hugr::{HugrView, IncomingPort, Node, OutgoingPort, PortIndex, Wire};
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
                insert_opaque_subgraph(qubits, bits, params, decoder, &payload)
            }
            PytketOperation {
                op_type: PytketOptype::CircBox,
                op_box: Some(OpBox::CircBox { id: _id, circuit }),
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
                let internal =
                    circuit.decode_inplace(decoder.builder.hugr_mut(), target, options)?;

                decoder
                    .wire_up_node(internal, qubits, qubits, bits, bits, params)
                    .map_err(|e| e.hugr_op("DFG"))?;

                Ok(DecodeStatus::Success)
            }
            _ => Ok(DecodeStatus::Unsupported),
        }
    }
}

/// Insert a subgraph encoded in the payload of a pytket barrier operation into
/// the Hugr being decoded.
///
/// This function involves accessing various internal definitions of `decoder`
/// to deal with wires between unsupported subgraphs.
fn insert_opaque_subgraph(
    qubits: &[TrackedQubit],
    bits: &[TrackedBit],
    params: &[LoadedParameter],
    decoder: &mut PytketDecoderContext<'_>,
    payload: &OpaqueSubgraphPayload,
) -> Result<DecodeStatus, PytketDecodeError> {
    let to_insert_hugr = decoder.get_opaque_subgraph(payload.typ())?;
    let to_insert_signature = to_insert_hugr.inner_function_type().unwrap();

    let module = decoder.builder.hugr().module_root();
    let region = decoder.builder.container_node();

    // Collect the non-IO nodes in the hugr we plan to insert.
    let Some([to_insert_input, to_insert_output]) =
        to_insert_hugr.get_io(to_insert_hugr.entrypoint())
    else {
        return Err(PytketDecodeError::custom(
            "Opaque subgraph payload has a non-dataflow parent as entrypoint",
        ));
    };
    let entrypoint_children = to_insert_hugr
        .children(to_insert_hugr.entrypoint())
        .filter(|c| *c != to_insert_input && *c != to_insert_output)
        .map(|c| (c, region));

    // Compute the inputs and output ports of the subgraph.
    // Since we insert the nodes inside the region directly (without the I/O nodes),
    // we need to do some bookkeeping to match the ports.
    //
    // `to_insert_inputs` is a vector of vectors, where each first-level entry corresponds to
    // an input to the subgraph / element in `payload.inputs`, and the second-level
    // list is all the target ports that connect to that input.
    //
    // `to_insert_outputs` is just a vector of node+outgoing ports.
    let to_insert_inputs: hugr::hugr::views::sibling_subgraph::IncomingPorts<Node> =
        to_insert_signature
            .input_ports()
            .map(|p| {
                to_insert_hugr
                    .linked_inputs(to_insert_input, p.index())
                    .collect_vec()
            })
            .collect_vec();
    let to_insert_outputs: Vec<(Node, OutgoingPort)> = to_insert_signature
        .output_ports()
        .map(|p| {
            to_insert_hugr
                .single_linked_output(to_insert_output, p.index())
                .unwrap()
        })
        .collect_vec();

    // Collect any module child that does not contain the entrypoint function.
    //
    // These are global functions or constant definitions.
    let entrypoint_function = std::iter::successors(Some(to_insert_hugr.entrypoint()), |&node| {
        let parent = to_insert_hugr.get_parent(node)?;
        if parent == module {
            None
        } else {
            Some(parent)
        }
    })
    .last()
    .unwrap();
    let module_children = to_insert_hugr
        .children(to_insert_hugr.module_root())
        .filter(|c| *c != entrypoint_function)
        .map(|c| (c, module));

    // Insert the hugr's entrypoint region directly into the region being built,
    // and any other function in the HUGR module into the module being built.
    let insertion_roots = entrypoint_children.chain(module_children).collect_vec();
    let insertion_result = decoder
        .builder
        .hugr_mut()
        .insert_forest(to_insert_hugr, insertion_roots)
        .unwrap_or_else(|e| panic!("Invalid `insertion_roots`. {e}"));

    // Gather and connect the wires between the previously decoded nodes and the
    // inserted subgraph inputs, using the types and edge IDs from the payload.
    let mut input_qubits = qubits;
    let mut input_bits = bits;
    let mut input_params = params;
    // A list of incoming ports corresponding to [`EncodedEdgeID`]s that must be
    // connected once the outgoing port is created.
    //
    // This handles the case where unsupported subgraphs in opaque barriers on
    // the pytket circuit get reordered and input ports are seen before their
    // outputs.
    let mut pending_encoded_edge_connections: HashMap<EncodedEdgeID, Vec<(Node, IncomingPort)>> =
        HashMap::new();
    for ((ty, edge_id), targets) in payload.inputs().zip_eq(to_insert_inputs) {
        let found_wire = decoder.wire_tracker.find_typed_wire(
            decoder.config(),
            ty,
            &mut input_qubits,
            &mut input_bits,
            &mut input_params,
            Some(edge_id),
        )?;

        let wire = match found_wire {
            FoundWire::Register(wire_data) => wire_data.wire(),
            FoundWire::Parameter(param) => param.wire(),
            FoundWire::Unsupported { id } => {
                let Some(wire) = decoder.wire_tracker.get_unsupported_wire(id) else {
                    // The corresponding outgoing port has not been created yet, so we
                    // register the edge id and the targets to be connected later.
                    pending_encoded_edge_connections
                        .entry(id)
                        .or_default()
                        .extend(targets);
                    continue;
                };
                *wire
            }
        };

        for (to_insert_node, port) in targets {
            let node = *insertion_result.node_map.get(&to_insert_node).unwrap();
            decoder
                .builder
                .hugr_mut()
                .connect(wire.node(), wire.source(), node, port);
        }
    }

    // Register the subgraph outputs in the wire tracker.
    let mut output_qubits = qubits;
    let mut output_bits = bits;
    for ((ty, edge_id), (unsupported_node, port)) in payload.outputs().zip_eq(to_insert_outputs) {
        let wire = Wire::new(unsupported_node, port);
        match decoder.config().type_to_pytket(ty) {
            Some(counts) => {
                let wire_qubits = split_off(&mut output_qubits, ..counts.qubits);
                let wire_bits = split_off(&mut output_bits, ..counts.bits);
                if wire_qubits.is_none() || wire_bits.is_none() {
                    return Err(make_unexpected_node_out_error(
                        decoder.config(),
                        payload.outputs().map(|(ty, _)| ty),
                        qubits.len(),
                        bits.len(),
                    ));
                }
                decoder.wire_tracker.track_wire(
                    wire,
                    Arc::new(ty.clone()),
                    wire_qubits.unwrap().iter().cloned(),
                    wire_bits.unwrap().iter().cloned(),
                )?;
            }
            None => {
                // This is an unsupported wire, so we just register the edge id to the wire.
                decoder
                    .wire_tracker
                    .register_unsupported_wire(edge_id, wire);
                // If we've registered a pending connection for this edge id, connect it now.
                if let Some(targets) = pending_encoded_edge_connections.remove(&edge_id) {
                    for (node, port) in targets {
                        decoder
                            .builder
                            .hugr_mut()
                            .connect(wire.node(), wire.source(), node, port);
                    }
                }
            }
        }
    }

    // Mark any unused qubits and bits as outdated.
    qubits.iter().for_each(|q| {
        decoder.wire_tracker.mark_qubit_outdated(q.clone());
    });
    bits.iter().for_each(|b| {
        decoder.wire_tracker.mark_bit_outdated(b.clone());
    });

    Ok(DecodeStatus::Success)
}

// TODO: Replace with array's `split_off` method once MSRV is ≥1.87
fn split_off<'a, T>(slice: &mut &'a [T], range: RangeTo<usize>) -> Option<&'a [T]> {
    let split_index = range.end;
    if split_index > slice.len() {
        return None;
    }
    let (front, back) = slice.split_at(split_index);
    *slice = back;
    Some(front)
}

/// Helper to compute the expected register counts before generating a
/// [`PytketDecodeErrorInner::UnexpectedNodeOutput`] error when registering the
/// outputs of an unsupported subgraph.
///
/// Processes all the output types to compute the number of qubits and bits we
/// required to have available.
fn make_unexpected_node_out_error<'ty>(
    config: &PytketDecoderConfig,
    output_types: impl IntoIterator<Item = &'ty Type>,
    available_qubits: usize,
    available_bits: usize,
) -> PytketDecodeError {
    let mut expected_count = RegisterCount::default();
    for ty in output_types {
        expected_count += config.type_to_pytket(ty).unwrap_or_default();
    }
    PytketDecodeErrorInner::UnexpectedNodeOutput {
        expected_qubits: expected_count.qubits,
        expected_bits: expected_count.bits,
        circ_qubits: available_qubits,
        circ_bits: available_bits,
    }
    .wrap()
}
