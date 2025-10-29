//! Methods to decode opaque subgraphs from a pytket barrier operation.

use std::collections::HashMap;
use std::ops::RangeTo;
use std::sync::Arc;

use hugr::builder::Container;
use hugr::hugr::hugrmut::HugrMut;
use hugr::types::Type;
use hugr::{Hugr, HugrView, IncomingPort, Node, OutgoingPort, PortIndex, Wire};
use hugr_core::hugr::internal::HugrMutInternals;
use itertools::Itertools;

use crate::serialize::pytket::decoder::{
    DecodeStatus, FoundWire, LoadedParameter, PytketDecoderContext, TrackedBit, TrackedQubit,
};
use crate::serialize::pytket::extension::RegisterCount;
use crate::serialize::pytket::opaque::{EncodedEdgeID, OpaqueSubgraphPayload, SubgraphId};
use crate::serialize::pytket::{PytketDecodeError, PytketDecodeErrorInner, PytketDecoderConfig};

impl<'h> PytketDecoderContext<'h> {
    /// Insert a subgraph encoded in the payload of a pytket barrier operation into
    /// the Hugr being decoded.
    ///
    /// This function involves accessing various internal definitions of `decoder`
    /// to deal with wires between unsupported subgraphs.
    pub(in crate::serialize::pytket) fn insert_subgraph_from_payload(
        &mut self,
        qubits: &[TrackedQubit],
        bits: &[TrackedBit],
        params: &[LoadedParameter],
        payload: &OpaqueSubgraphPayload,
    ) -> Result<DecodeStatus, PytketDecodeError> {
        match payload {
            OpaqueSubgraphPayload::External { id } => {
                self.insert_external_subgraph(*id, qubits, bits, params)
            }
            OpaqueSubgraphPayload::Inline {
                hugr_envelope,
                inputs,
                outputs,
            } => self.insert_inline_subgraph(hugr_envelope, inputs, outputs, qubits, bits, params),
        }
    }

    /// Move the subgraph nodes referenced by an
    /// [`OpaqueSubgraphPayload::External`] into the region being decoded.
    fn insert_external_subgraph(
        &mut self,
        id: SubgraphId,
        qubits: &[TrackedQubit],
        bits: &[TrackedBit],
        params: &[LoadedParameter],
    ) -> Result<DecodeStatus, PytketDecodeError> {
        fn mk_subgraph_error(id: SubgraphId, context: impl ToString) -> PytketDecodeError {
            PytketDecodeErrorInner::InvalidExternalSubgraph {
                id,
                context: context.to_string(),
            }
            .wrap()
        }

        let Some(subgraph) = self.opaque_subgraphs.and_then(|s| s.get(id)) else {
            return Err(PytketDecodeErrorInner::OpaqueSubgraphNotFound { id }.wrap());
        };
        let signature = subgraph.signature(self.builder.hugr());

        let old_parent = self
            .builder
            .hugr()
            .get_parent(subgraph.nodes()[0])
            .ok_or_else(|| mk_subgraph_error(id, "Subgraph must contain be dataflow nodes."))?;
        let [old_input, old_output] = self.builder.hugr().get_io(old_parent).ok_or_else(|| {
            mk_subgraph_error(id, "Stored subgraph must be in a dataflow region.")
        })?;
        let new_parent = self.builder.container_node();

        // Re-parent the nodes in the subgraph.
        for &node in subgraph.nodes() {
            self.builder.hugr_mut().set_parent(node, new_parent);
        }

        // Re-wire the input wires that should be connected to nodes in the new region.
        let mut input_qubits = qubits;
        let mut input_bits = bits;
        let mut input_params = params;
        for (ty, targets) in signature.input().iter().zip_eq(subgraph.incoming_ports()) {
            let wire = match self.wire_tracker.find_typed_wire(
                self.config(),
                ty,
                &mut input_qubits,
                &mut input_bits,
                &mut input_params,
                None,
            ) {
                Ok(FoundWire::Register(wire_data)) => wire_data.wire(),
                Ok(FoundWire::Parameter(param)) => param.wire(),
                Ok(FoundWire::Unsupported { .. }) => {
                    unreachable!("`unsupported_wire` not passed to `find_typed_wire`.");
                }
                Err(PytketDecodeError {
                    inner: PytketDecodeErrorInner::NoMatchingWire { .. },
                    ..
                }) => {
                    // Not a qubit or bit wire.
                    // If it was linked to the old circuit, we need to re-wire it.
                    // Otherwise we just leave it connected to the node it was linked to.
                    let Some((neigh, neigh_port)) = targets.first().and_then(|(tgt, port)| {
                        self.builder.hugr().single_linked_output(*tgt, *port)
                    }) else {
                        continue;
                    };
                    if neigh != old_input {
                        continue;
                    }
                    Wire::new(neigh, neigh_port)
                }
                Err(e) => return Err(e),
            };

            for (tgt, port) in targets {
                self.builder
                    .hugr_mut()
                    .connect(wire.node(), wire.source(), *tgt, *port);
            }
        }

        // Register the output wires that should be connected to nodes in the new region.
        //
        // Re-wire wires from the subgraph to the old region's outputs.
        let mut output_qubits = qubits;
        let mut output_bits = bits;
        for (ty, (src, src_port)) in signature.input().iter().zip_eq(subgraph.outgoing_ports()) {
            let wire = Wire::new(*src, *src_port);
            match self.config().type_to_pytket(ty) {
                Some(counts) => {
                    // Make sure to disconnect the old wire.
                    self.builder.hugr_mut().disconnect(*src, *src_port);

                    let wire_qubits = split_off(&mut output_qubits, ..counts.qubits);
                    let wire_bits = split_off(&mut output_bits, ..counts.bits);
                    if wire_qubits.is_none() || wire_bits.is_none() {
                        return Err(make_unexpected_node_out_error(
                            self.config(),
                            signature.output().iter(),
                            qubits.len(),
                            bits.len(),
                        ));
                    }
                    self.wire_tracker.track_wire(
                        wire,
                        Arc::new(ty.clone()),
                        wire_qubits.unwrap().iter().cloned(),
                        wire_bits.unwrap().iter().cloned(),
                    )?;
                }
                None => {
                    // This is an unsupported wire.
                    // If it was connected to the old region output, rewire it.
                    // Otherwise leave it connected.
                    for (tgt, tgt_port) in self
                        .builder
                        .hugr()
                        .linked_inputs(*src, *src_port)
                        .collect_vec()
                    {
                        if tgt == old_output {
                            self.builder.hugr_mut().disconnect(tgt, tgt_port);
                            self.builder
                                .hugr_mut()
                                .connect(*src, *src_port, tgt, tgt_port);
                        }
                    }
                }
            }
        }

        // Mark the used qubits and bits as outdated.
        qubits.iter().for_each(|q| {
            self.wire_tracker.mark_qubit_outdated(q.clone());
        });
        bits.iter().for_each(|b| {
            self.wire_tracker.mark_bit_outdated(b.clone());
        });

        Ok(DecodeStatus::Success)
    }

    /// Insert an [`OpaqueSubgraphPayload::Inline`] into the Hugr being decoded.
    fn insert_inline_subgraph(
        &mut self,
        hugr_envelope: &str,
        payload_inputs: &[(Type, EncodedEdgeID)],
        payload_outputs: &[(Type, EncodedEdgeID)],
        qubits: &[TrackedQubit],
        bits: &[TrackedBit],
        params: &[LoadedParameter],
    ) -> Result<DecodeStatus, PytketDecodeError> {
        let to_insert_hugr = Hugr::load_str(hugr_envelope, Some(self.options.extension_registry()))
            .map_err(|e| PytketDecodeErrorInner::UnsupportedSubgraphPayload { source: e })?;
        let to_insert_signature = to_insert_hugr.inner_function_type().unwrap();

        let module = self.builder.hugr().module_root();
        let region = self.builder.container_node();

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
        let entrypoint_function =
            std::iter::successors(Some(to_insert_hugr.entrypoint()), |&node| {
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
        let insertion_result = self
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
        let mut pending_encoded_edge_connections: HashMap<
            EncodedEdgeID,
            Vec<(Node, IncomingPort)>,
        > = HashMap::new();
        for ((ty, edge_id), targets) in payload_inputs.iter().zip_eq(to_insert_inputs) {
            let found_wire = self.wire_tracker.find_typed_wire(
                self.config(),
                ty,
                &mut input_qubits,
                &mut input_bits,
                &mut input_params,
                Some(*edge_id),
            )?;

            let wire = match found_wire {
                FoundWire::Register(wire_data) => wire_data.wire(),
                FoundWire::Parameter(param) => param.wire(),
                FoundWire::Unsupported { id } => {
                    let Some(wire) = self.wire_tracker.get_unsupported_wire(id) else {
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
                self.builder
                    .hugr_mut()
                    .connect(wire.node(), wire.source(), node, port);
            }
        }

        // Register the subgraph outputs in the wire tracker.
        let mut output_qubits = qubits;
        let mut output_bits = bits;
        for ((ty, edge_id), (unsupported_node, port)) in
            payload_outputs.iter().zip_eq(to_insert_outputs)
        {
            let wire = Wire::new(unsupported_node, port);
            match self.config().type_to_pytket(ty) {
                Some(counts) => {
                    let wire_qubits = split_off(&mut output_qubits, ..counts.qubits);
                    let wire_bits = split_off(&mut output_bits, ..counts.bits);
                    if wire_qubits.is_none() || wire_bits.is_none() {
                        return Err(make_unexpected_node_out_error(
                            self.config(),
                            payload_outputs.iter().map(|(ty, _)| ty),
                            qubits.len(),
                            bits.len(),
                        ));
                    }
                    self.wire_tracker.track_wire(
                        wire,
                        Arc::new(ty.clone()),
                        wire_qubits.unwrap().iter().cloned(),
                        wire_bits.unwrap().iter().cloned(),
                    )?;
                }
                None => {
                    // This is an unsupported wire, so we just register the edge id to the wire.
                    self.wire_tracker.register_unsupported_wire(*edge_id, wire);
                    // If we've registered a pending connection for this edge id, connect it now.
                    if let Some(targets) = pending_encoded_edge_connections.remove(edge_id) {
                        for (node, port) in targets {
                            self.builder
                                .hugr_mut()
                                .connect(wire.node(), wire.source(), node, port);
                        }
                    }
                }
            }
        }

        // Mark the used qubits and bits as outdated.
        qubits.iter().for_each(|q| {
            self.wire_tracker.mark_qubit_outdated(q.clone());
        });
        bits.iter().for_each(|b| {
            self.wire_tracker.mark_bit_outdated(b.clone());
        });

        Ok(DecodeStatus::Success)
    }
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
