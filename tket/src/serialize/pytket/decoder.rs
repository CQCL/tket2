//! Intermediate structure for decoding [`SerialCircuit`]s into [`Hugr`]s.

mod param;
mod tracked_elem;
mod wires;

use hugr::hugr::hugrmut::HugrMut;
pub use param::{LoadedParameter, ParameterType};
pub use tracked_elem::{TrackedBit, TrackedQubit};
pub use wires::TrackedWires;

pub(super) use wires::FindTypedWireResult;

use std::sync::Arc;

use hugr::builder::{
    BuildHandle, Container, DFGBuilder, Dataflow, DataflowSubContainer, FunctionBuilder,
};
use hugr::extension::prelude::{bool_t, qb_t};
use hugr::ops::handle::{DataflowOpID, NodeHandle};
use hugr::ops::{OpParent, OpTrait, OpType, Value, DFG};
use hugr::types::{Signature, Type, TypeRow};
use hugr::{Hugr, HugrView, Node, OutgoingPort, Wire};
use tracked_elem::{TrackedBitId, TrackedQubitId};

use itertools::Itertools;
use serde_json::json;
use tket_json_rs::circuit_json;
use tket_json_rs::circuit_json::SerialCircuit;

use super::{
    PytketDecodeError, METADATA_B_REGISTERS, METADATA_INPUT_PARAMETERS, METADATA_PHASE,
    METADATA_Q_REGISTERS,
};
use crate::extension::rotation::rotation_type;
use crate::serialize::pytket::config::PytketDecoderConfig;
use crate::serialize::pytket::decoder::wires::WireTracker;
use crate::serialize::pytket::extension::{build_opaque_tket_op, RegisterCount};
use crate::serialize::pytket::opaque::{OpaqueSubgraphPayloadType, OpaqueSubgraphs};
use crate::serialize::pytket::{DecodeInsertionTarget, DecodeOptions, PytketDecodeErrorInner};
use crate::TketOp;

/// State of the tket circuit being decoded.
///
/// The state of an in-progress [`FunctionBuilder`] being built from a
/// [`SerialCircuit`].
///
/// The generic parameter `H` is the HugrView type of the Hugr that was encoded
/// into the circuit, if any. This is required when the encoded pytket circuit
/// contains opaque barriers that reference subgraphs in the original HUGR. See
/// [`OpaqueSubgraphPayload`][super::opaque::OpaqueSubgraphPayload] for more details.
#[derive(Debug)]
pub struct PytketDecoderContext<'h> {
    /// The Hugr being built.
    pub builder: DFGBuilder<&'h mut Hugr>,
    /// A tracker keeping track of the generated wires and their corresponding types.
    pub(super) wire_tracker: Box<WireTracker>,
    /// Options used when decoding the circuit.
    ///
    /// `DecodeOptions::config` is
    options: DecodeOptions,
    /// A registry of opaque subgraphs from `original_hugr`, that are referenced by opaque barriers in the pytket circuit
    /// via their [`SubgraphId`].
    opaque_subgraphs: Option<&'h OpaqueSubgraphs<Node>>,
}

impl<'h> PytketDecoderContext<'h> {
    /// Initialize a new [`PytketDecoderContext`], using the metadata from a
    /// [`SerialCircuit`].
    ///
    /// The circuit will be defined as a new function in the HUGR.
    ///
    /// # Arguments
    ///
    /// - `serialcirc`: The serialised circuit to decode.
    /// - `hugr`: The [`Hugr`] to define the new function in.
    /// - `fn_name`: The name of the function to create. If `None`, we will use
    ///   the name of the circuit, or "main" if the circuit has no name.
    /// - `signature`: The signature of the function to create. If `None`, we
    ///   will use qubits and bools.
    /// - `input_params`: A list of parameter names to add to the function
    ///   input. If additional parameters are found in the circuit, they will be
    ///   added after these.
    /// - `config`: The configuration for the decoder, containing custom
    ///   operation decoders.
    ///
    /// # Defining the function signature
    ///
    /// If `signature` is not provided, we default to a sequence of qubit types
    /// followed by bool types, according to the qubit and bit counts in the
    /// circuit.
    ///
    /// If provided, we produce a hugr with the given signature instead. The
    /// amount of qubit and bit registers in the `serialcirc` must match the
    /// count in the signature input types, as defined by the type translators
    /// in the [`PytketDecoderConfig`].
    ///
    /// The signature may include bare parameter wires (e.g. `float64` or
    /// `rotation`) mixed between the value types. These will be associated with
    /// the `input_params` names if possible. Any remaining parameter in
    /// `input_params` will be added as additional inputs with type
    /// [`rotation_type`]. Additional parameter inputs may be added during
    /// runtime, as new free variables are found in the command arguments.
    pub(super) fn new(
        serialcirc: &SerialCircuit,
        hugr: &'h mut Hugr,
        target: DecodeInsertionTarget,
        mut options: DecodeOptions,
    ) -> Result<Self, PytketDecodeError> {
        // Ensure that the set of decoders is present, use a default one if not.
        if options.config.is_none() {
            options.with_default_config();
        }

        // Compute the signature of the decoded region, if not provided, and
        // initialize the DFG builder.
        let signature = options.signature.clone().unwrap_or_else(|| {
            let num_qubits = serialcirc.qubits.len();
            let num_bits = serialcirc.bits.len();
            let types: TypeRow = [vec![qb_t(); num_qubits], vec![bool_t(); num_bits]]
                .concat()
                .into();
            Signature::new(types.clone(), types)
        });
        let mut dfg: DFGBuilder<&mut Hugr> = match target {
            DecodeInsertionTarget::Function { fn_name } => {
                let name = fn_name
                    .or_else(|| serialcirc.name.clone())
                    .unwrap_or_default();
                FunctionBuilder::with_hugr(hugr, name, signature.clone())
                    .unwrap()
                    .into_dfg_builder()
            }
            DecodeInsertionTarget::Region { parent } => {
                let op = DFG {
                    signature: signature.clone(),
                };
                let dfg = hugr.add_node_with_parent(parent, op);
                DFGBuilder::create_with_io(hugr, dfg, signature.clone()).unwrap()
            }
        };

        Self::init_metadata(&mut dfg, serialcirc);
        let wire_tracker = Self::init_wire_tracker(
            serialcirc,
            &mut dfg,
            &signature.input,
            options.input_params.iter().cloned(),
            options.get_config(),
        )?;

        if !serialcirc.phase.is_empty() {
            // TODO - add a phase gate
            // <https://github.com/CQCL/tket2/issues/598>
            // let phase = Param::new(serialcirc.phase);
            // decoder.add_phase(phase);
        }

        Ok(PytketDecoderContext {
            builder: dfg,
            wire_tracker: Box::new(wire_tracker),
            options,
            opaque_subgraphs: None,
        })
    }

    /// Store the serialised circuit information as HUGR metadata,
    /// so it can be reused later when re-encoding the circuit.
    fn init_metadata(dfg: &mut DFGBuilder<&mut Hugr>, serialcirc: &SerialCircuit) {
        // Metadata. The circuit requires "name", and we store other things that
        // should pass through the serialization roundtrip.
        dfg.set_metadata(METADATA_PHASE, json!(serialcirc.phase));
        dfg.set_metadata(METADATA_Q_REGISTERS, json!(serialcirc.qubits));
        dfg.set_metadata(METADATA_B_REGISTERS, json!(serialcirc.bits));
    }

    /// Initialize the wire tracker with the input wires.
    ///
    /// Checks that the signature matches the expected number of qubits, bits,
    /// and parameters. See
    /// [`DecodeOptions::signature`][super::DecodeOptions::signature] for more
    /// details.
    ///
    /// Utility method for [`PytketDecoderContext::new`].
    ///
    /// # Panics
    ///
    /// If the dfg builder does not support adding input wires.
    /// (That is, we're not building a FuncDefn or a DFG).
    fn init_wire_tracker(
        serialcirc: &SerialCircuit,
        dfg: &mut DFGBuilder<&mut Hugr>,
        input_types: &TypeRow,
        input_params: impl IntoIterator<Item = String>,
        config: &PytketDecoderConfig,
    ) -> Result<WireTracker, PytketDecodeError> {
        let num_qubits = serialcirc.qubits.len();
        let num_bits = serialcirc.bits.len();
        let mut input_params = input_params.into_iter();

        let mut wire_tracker = WireTracker::with_capacity(num_qubits, num_bits);

        let mut qubits = serialcirc
            .qubits
            .iter()
            .map(|qb| TrackedQubit::new(TrackedQubitId(0), Arc::new(qb.id.clone())));
        let mut bits = serialcirc
            .bits
            .iter()
            .map(|bit| TrackedBit::new(TrackedBitId(0), Arc::new(bit.id.clone())));
        let mut added_inputs_count = RegisterCount::default();
        for (wire, ty) in dfg.input_wires().zip(input_types.iter()) {
            let elem_counts = config.type_to_pytket(ty).unwrap_or_default();

            if elem_counts.is_empty() {
                // Input is ignored.
                continue;
            }

            let wire_qubits = qubits.by_ref().take(elem_counts.qubits).collect_vec();
            let wire_bits = bits.by_ref().take(elem_counts.bits).collect_vec();
            // Error out if the signature inputs has more qubits than the ones defined in the circuit.
            // We ignore additional bits, and leave them disconnected.
            if wire_qubits.len() != elem_counts.qubits {
                let expected_count: RegisterCount = input_types
                    .iter()
                    .map(|t| config.type_to_pytket(t).unwrap_or_default())
                    .sum();
                return Err(PytketDecodeErrorInner::InvalidInputSignature {
                    input_types: input_types.iter().map(|t| t.to_string()).collect(),
                    expected_qubits: expected_count.qubits,
                    expected_bits: expected_count.bits,
                    circ_qubits: serialcirc.qubits.len(),
                    circ_bits: serialcirc.bits.len(),
                }
                .wrap());
            }
            wire_tracker.track_wire(wire, Arc::new(ty.clone()), wire_qubits, wire_bits)?;

            if elem_counts.params > 0 {
                if elem_counts != RegisterCount::only_params(1) {
                    return Err(PytketDecodeErrorInner::UnsupportedParametersInInput {
                        ty: ty.to_string(),
                        num_params: elem_counts.params,
                    }
                    .wrap());
                }
                let loaded = if ty == &rotation_type() {
                    LoadedParameter::rotation(wire)
                } else {
                    LoadedParameter::float_half_turns(wire)
                };
                match input_params.next() {
                    Some(param) => wire_tracker.register_input_parameter(loaded, param)?,
                    None => wire_tracker.register_unused_parameter_input(loaded),
                };
            }

            added_inputs_count += elem_counts;
        }

        // Insert any remaining named parameters as new inputs
        for param in input_params {
            let wire = dfg
                .add_input(rotation_type())
                .expect("Must be building a FuncDefn or a DFG");
            wire_tracker.register_input_parameter(LoadedParameter::rotation(wire), param)?;
        }

        // Any additional qubits or bits required by the circuit get initialized to |0> / false.
        for q in qubits {
            let q_wire = dfg.add_dataflow_op(TketOp::QAlloc, []).unwrap().out_wire(0);
            wire_tracker.track_wire(q_wire, q.ty(), [q], [])?;
        }
        for b in bits {
            let b_wire = dfg.add_load_value(Value::false_val());
            wire_tracker.track_wire(b_wire, b.ty(), [], [b])?;
        }

        wire_tracker.compute_output_permutation(&serialcirc.implicit_permutation);

        Ok(wire_tracker)
    }

    /// Finish building the function definition for the legacy tket circuit.
    ///
    /// After this call, the HUGR passed to the [`PytketDecoderContext::new`]
    /// constructor will contain the fully defined function.
    ///
    /// The original Hugr entrypoint is _not_ modified, it must be set by the
    /// caller if required.
    pub(super) fn finish(mut self) -> Result<Node, PytketDecodeError> {
        // Order the final wires according to the serial circuit register order.
        let qubits = self
            .wire_tracker
            .known_pytket_qubits()
            .cloned()
            .collect_vec();
        let bits = self.wire_tracker.known_pytket_bits().cloned().collect_vec();
        let function_type = self
            .builder
            .hugr()
            .get_optype(self.builder.container_node())
            .inner_function_type()
            .unwrap();
        let expected_output_types = function_type.output_types().iter().cloned().collect_vec();

        let output_wires = self
            .find_typed_wires(&expected_output_types, &qubits, &bits, &[])
            .map_err(|e| e.hugr_op("Output"))?;

        output_wires
            .check_types(expected_output_types.as_slice(), 0)
            .map_err(|mut e| {
                if let PytketDecodeError {
                    inner:
                        PytketDecodeErrorInner::UnexpectedInputWires {
                            expected_types,
                            actual_types,
                            ..
                        },
                    ..
                } = e
                {
                    e.inner = PytketDecodeErrorInner::InvalidOutputSignature {
                        expected_types: expected_types.unwrap_or_default(),
                        actual_types: actual_types.unwrap_or_default(),
                    };
                };
                e.hugr_op("Output")
            })?;
        let output_wires = output_wires.wires();

        // Store the name for the input parameter wires
        let input_params = self.wire_tracker.finish();
        if !input_params.is_empty() {
            self.builder.set_metadata(
                METADATA_INPUT_PARAMETERS,
                json!(input_params.into_iter().collect_vec()),
            );
        }

        Ok(self
            .builder
            .finish_with_outputs(output_wires)
            .map_err(PytketDecodeError::custom)?
            .node())
    }

    /// Register the set of opaque subgraphs that are present in the HUGR being decoded.
    ///
    /// # Arguments
    /// - `opaque_subgraphs`: A registry of opaque subgraphs from
    ///   `self.builder.hugr()`, that are referenced by opaque barriers in the
    ///   pytket circuit via their [`SubgraphId`].
    pub(super) fn register_opaque_subgraphs(
        &mut self,
        opaque_subgraphs: &'h OpaqueSubgraphs<Node>,
    ) {
        self.opaque_subgraphs = Some(opaque_subgraphs);
    }

    /// Decode a list of pytket commands.
    pub(super) fn run_decoder(
        &mut self,
        commands: &[circuit_json::Command],
    ) -> Result<(), PytketDecodeError> {
        let config = self.config().clone();
        for com in commands {
            let op_type = com.op.op_type;
            self.process_command(com, config.as_ref())
                .map_err(|e| e.pytket_op(&op_type))?;
        }
        Ok(())
    }

    /// Add a tket1 [`circuit_json::Command`] from the serial circuit to the
    /// decoder.
    pub(super) fn process_command(
        &mut self,
        command: &circuit_json::Command,
        config: &PytketDecoderConfig,
    ) -> Result<(), PytketDecodeError> {
        let circuit_json::Command { op, args, opgroup } = command;

        // Find the latest [`TrackedQubit`] and [`TrackedBit`] for the command registers.
        let (qubits, bits) = self.wire_tracker.pytket_args_to_tracked_elems(args)?;

        // Collect the parameters used in the command.
        let params: Vec<LoadedParameter> = match &op.params {
            Some(params) => params
                .iter()
                .map(|v| self.load_half_turns(v.as_str()))
                .collect_vec(),
            None => Vec::new(),
        };

        // Try to decode the command with the configured decoders.
        match config.op_to_hugr(op, &qubits, &bits, &params, opgroup, self)? {
            DecodeStatus::Success => {}
            DecodeStatus::Unsupported => {
                // The command couldn't be translated into a native HUGR counterpart, so
                // we generate an opaque `Tk1Op` instead.
                build_opaque_tket_op(op, &qubits, &bits, &params, opgroup, self)?;
            }
        }
        Ok(())
    }

    /// Returns a tracked opaque subgraph encoded in an opaque barrier in the pytket circuit.
    ///
    /// See [`OpaqueSubgraphPayload`][super::opaque::OpaqueSubgraphPayload]
    /// for more details.
    pub(super) fn get_opaque_subgraph(
        &self,
        payload: &OpaqueSubgraphPayloadType,
    ) -> Result<Hugr, PytketDecodeError> {
        match payload {
            OpaqueSubgraphPayloadType::Inline { hugr_envelope } => {
                let hugr = Hugr::load_str(hugr_envelope, Some(self.options.extension_registry()))
                    .map_err(|e| PytketDecodeErrorInner::UnsupportedSubgraphPayload {
                    source: e,
                })?;
                Ok(hugr)
            }
            OpaqueSubgraphPayloadType::External { id } => match self.opaque_subgraphs {
                Some(subgraphs) if subgraphs.contains(*id) => {
                    let hugr = subgraphs[*id].extract_subgraph(self.builder.hugr(), id.to_string());
                    Ok(hugr)
                }
                _ => Err(PytketDecodeErrorInner::OpaqueSubgraphNotFound { id: *id }.wrap()),
            },
        }
    }
}

/// Public API, used by the [`PytketDecoder`][super::extension::PytketDecoder] implementers.
impl<'h> PytketDecoderContext<'h> {
    /// Returns a new set of [TrackedWires] for a list of [`TrackedQubit`]s,
    /// [`TrackedBit`]s, and [`LoadedParameter`]s following the required types.
    ///
    /// Returns an error if a valid set of wires with the given types cannot be
    /// found.
    ///
    /// The qubit and bit arguments are only consumed as required by the types.
    /// Some registers may be left unused.
    ///
    /// # Arguments
    ///
    /// * `config` - The configuration for the decoder, used to count the qubits and bits required by each type.
    /// * `hugr` - The hugr to load the parameters to.
    /// * `types` - The types of the arguments we require in the wires.
    /// * `qubit_args` - The list of tracked qubits we require in the wires.
    /// * `bit_args` - The list of tracked bits we require in the wire.
    /// * `params` - The list of parameters to load to wires. See
    ///   [`PytketDecoderContext::load_half_turns`] for more details.
    ///
    /// # Errors
    ///
    /// - [`PytketDecodeErrorInner::OutdatedQubit`] if a qubit in `qubit_args` was marked as outdated.
    /// - [`PytketDecodeErrorInner::OutdatedBit`] if a bit in `bit_args` was marked as outdated.
    /// - [`PytketDecodeErrorInner::UnexpectedInputType`] if a type in `types` cannot be mapped to a [`RegisterCount`]
    /// - [`PytketDecodeErrorInner::NoMatchingWire`] if there is no wire with the requested type for the given qubit/bit arguments.
    pub fn find_typed_wires(
        &self,
        types: &[Type],
        qubit_args: &[TrackedQubit],
        bit_args: &[TrackedBit],
        params: &[LoadedParameter],
    ) -> Result<TrackedWires, PytketDecodeError> {
        self.wire_tracker
            .find_typed_wires(self.config(), types, qubit_args, bit_args, params)
    }

    /// Connects the input ports of a node using a list of input qubits, bits,
    /// and pytket parameters. Registers the node's output wires in the wire
    /// tracker.
    ///
    /// The qubits registers in `wires` are reused between the operation inputs
    /// and outputs. Bit registers, on the other hand, are not reused. We use
    /// the first registers in `wires` for the bit inputs and the remaining
    /// registers for the outputs.
    ///
    /// The input wire types must match the operation's input signature, no type
    /// conversion is performed.
    ///
    /// The caller must take care of converting the parameter wires to the
    /// required types and units expected by the operation. An error will be
    /// returned if the parameter does not match the expected wire type, but the
    /// unit (radians or half-turns) cannot be checked automatically.
    ///
    /// Use [`Self::add_node_with_wires`] to insert a new node before wiring it
    /// up.
    ///
    /// # Arguments
    ///
    /// - `node`: The node to wire up.
    /// - `input_qubits`: The qubits to use as input.
    ///   This list must match exactly the number of input qubits required by the operation.
    /// - `output_qubits`: The qubits to use as output.
    ///   This list must match exactly the number of output qubits required by the operation.
    /// - `input_bits`: The bits to use as input.
    ///   These should match exactly the number of input bits required by the operation.
    /// - `output_bits`: The bits to use as output.
    ///   These should match exactly the number of output bits required by the operation.
    /// - `params`: The parameters to use for the operation inputs.
    ///   This should match exactly the number of input parameters required by the operation.
    ///
    /// # Errors
    ///
    /// - [`PytketDecodeErrorInner::NotEnoughInputRegisters`] if the register
    ///   count required to encode the node inputs does not match the ones provided.
    /// - [`PytketDecodeErrorInner::NotEnoughOutputRegisters`] if the register
    ///   count required to encode the node outputs does not match the ones provided.
    /// - [`PytketDecodeErrorInner::OutdatedQubit`] if a qubit in `qubits` was marked as outdated.
    /// - [`PytketDecodeErrorInner::OutdatedBit`] if a bit in `bits` was marked as outdated.
    /// - [`PytketDecodeErrorInner::UnexpectedInputType`] if a type in the node's signature cannot be mapped to a [`RegisterCount`]
    pub fn wire_up_node(
        &mut self,
        node: Node,
        input_qubits: &[TrackedQubit],
        output_qubits: &[TrackedQubit],
        input_bits: &[TrackedBit],
        output_bits: &[TrackedBit],
        params: &[LoadedParameter],
    ) -> Result<(), PytketDecodeError> {
        let Some(sig) = self.builder.hugr().signature(node) else {
            return Err(PytketDecodeError::custom(
                "Cannot wire up non-dataflow operation",
            ));
        };

        // Compute the amount of elements required by the operation,
        // and the amount of elements in the input wires.
        let op_input_count: RegisterCount = sig
            .input_types()
            .iter()
            .map(|ty| self.config().type_to_pytket(ty).unwrap_or_default())
            .sum();
        let op_output_count: RegisterCount = sig
            .output_types()
            .iter()
            .map(|ty| self.config().type_to_pytket(ty).unwrap_or_default())
            .sum();

        // Validate input counts
        if op_input_count.qubits != input_qubits.len()
            || op_input_count.bits != input_bits.len()
            || op_input_count.params != params.len()
        {
            let expected_types = sig
                .input_types()
                .iter()
                .map(ToString::to_string)
                .collect_vec();
            return Err(PytketDecodeErrorInner::NotEnoughInputRegisters {
                expected_types,
                expected_count: op_input_count,
                actual_count: RegisterCount::new(
                    input_qubits.len(),
                    input_bits.len(),
                    params.len(),
                ),
            }
            .wrap());
        }

        // Validate output counts
        // We currently don't allow output parameters
        if op_output_count.qubits != output_qubits.len()
            || op_output_count.bits != output_bits.len()
            || op_output_count.params != 0
        {
            let expected_types = sig
                .output_types()
                .iter()
                .map(ToString::to_string)
                .collect_vec();
            return Err(PytketDecodeErrorInner::NotEnoughOutputRegisters {
                expected_types,
                expected_count: op_output_count,
                actual_count: RegisterCount::new(output_qubits.len(), output_bits.len(), 0),
            }
            .wrap());
        }

        // Gather the input wires, with the types needed by the operation.
        let input_wires =
            self.find_typed_wires(sig.input_types(), input_qubits, input_bits, params)?;
        debug_assert_eq!(op_input_count, input_wires.register_count());

        for (input_idx, wire) in input_wires.wires().enumerate() {
            self.builder
                .hugr_mut()
                .connect(wire.node(), wire.source(), node, input_idx);
        }
        input_bits.iter().take(op_input_count.bits).for_each(|b| {
            self.wire_tracker.mark_bit_outdated(b.clone());
        });

        // Register the output wires.
        let output_qubits = output_qubits.iter().take(op_output_count.qubits).cloned();
        let output_bits = output_bits.iter().cloned();
        self.register_node_outputs(node.node(), output_qubits, output_bits)?;

        Ok(())
    }

    /// Add a new node to the HUGR and wire it up using the provided wire set as
    /// input and output.
    ///
    /// Inserts the new node into the HUGR, connects its input ports and
    /// registers the node's output wires in the wire tracker.
    ///
    /// See [`PytketDecoderContext::wire_up_node`] for more details on the
    /// wiring process.
    ///
    /// # Arguments
    ///
    /// - `op`: The operation to add.
    /// - `input_qubits`: The qubits to use as input.
    ///   This list must match exactly the number of input qubits required by the operation.
    /// - `output_qubits`: The qubits to use as output.
    ///   This list must match exactly the number of output qubits required by the operation.
    /// - `input_bits`: The bits to use as input.
    ///   These should match exactly the number of input bits required by the operation.
    /// - `output_bits`: The bits to use as output.
    ///   These should match exactly the number of output bits required by the operation.
    /// - `params`: The parameters to use for the operation inputs.
    ///   This should match exactly the number of input parameters required by the operation.
    ///
    /// # Errors
    ///
    /// See [`PytketDecoderContext::wire_up_node`] for error details.
    pub fn add_node_with_wires(
        &mut self,
        op: impl Into<OpType>,
        input_qubits: &[TrackedQubit],
        output_qubits: &[TrackedQubit],
        input_bits: &[TrackedBit],
        output_bits: &[TrackedBit],
        params: &[LoadedParameter],
    ) -> Result<BuildHandle<DataflowOpID>, PytketDecodeError> {
        let op: OpType = op.into();
        let op_name = op.to_string();
        let num_outputs = op
            .dataflow_signature()
            .map(|s| s.output_count())
            .unwrap_or_default();

        // Add the node to the HUGR.
        let node = self.builder.add_child_node(op);

        self.wire_up_node(
            node.node(),
            input_qubits,
            output_qubits,
            input_bits,
            output_bits,
            params,
        )
        .map_err(|e| e.hugr_op(&op_name))?;

        Ok((node, num_outputs).into())
    }

    /// Given a new node in the HUGR, register all of its output wires in the
    /// tracker.
    ///
    /// Consumes the bits and qubits in order. Any unused bits and qubits are
    /// marked as outdated, as they are assumed to have been consumed in the
    /// inputs.
    pub fn register_node_outputs(
        &mut self,
        node: Node,
        qubits: impl IntoIterator<Item = TrackedQubit>,
        bits: impl IntoIterator<Item = TrackedBit>,
    ) -> Result<(), PytketDecodeError> {
        let mut qubits = qubits.into_iter();
        let mut bits = bits.into_iter();
        let Some(sig) = self.builder.hugr().signature(node) else {
            return Err(PytketDecodeErrorInner::UnexpectedNodeOutput {
                expected_qubits: qubits.count(),
                expected_bits: bits.count(),
                circ_qubits: 0,
                circ_bits: 0,
            }
            .wrap());
        };

        let mut reg_count = RegisterCount::default();
        let mut port_types = sig.output_ports().zip(sig.output_types().iter());
        while let Some((port, ty)) = port_types.next() {
            let wire = Wire::new(node, port);
            let counts = self.config().type_to_pytket(ty).unwrap_or_default();
            reg_count += counts;

            // Get the qubits and bits for this wire.
            let wire_qubits = qubits.by_ref().take(counts.qubits).collect_vec();
            let wire_bits = bits.by_ref().take(counts.bits).collect_vec();
            if wire_qubits.len() != counts.qubits || wire_bits.len() != counts.bits {
                let expected_qubits = reg_count.qubits - counts.qubits + wire_qubits.len();
                let expected_bits = reg_count.bits - counts.bits + wire_bits.len();
                return Err(make_unexpected_node_out_error(
                    self.config(),
                    port_types,
                    reg_count,
                    expected_qubits,
                    expected_bits,
                ));
            }

            self.wire_tracker
                .track_wire(wire, Arc::new(ty.clone()), wire_qubits, wire_bits)?;
        }

        // Mark any unused qubits and bits as outdated.
        qubits.for_each(|q| {
            self.wire_tracker.mark_qubit_outdated(q);
        });
        bits.for_each(|b| {
            self.wire_tracker.mark_bit_outdated(b);
        });

        Ok(())
    }

    /// Loads a half-turns expression as a [`LoadedParameter`] in the hugr.
    ///
    /// - If the parameter is a known algebraic operation, adds the required op and recurses on its inputs.
    /// - If the parameter is a constant, a constant definition is added to the Hugr.
    /// - If the parameter is a variable, adds a new `rotation` input to the region.
    /// - If the parameter is a sympy expressions, adds it as a [`SympyOpDef`][crate::extension::sympy::SympyOpDef] black box.
    pub fn load_half_turns(&mut self, param: &str) -> LoadedParameter {
        self.wire_tracker
            .load_half_turns_parameter(&mut self.builder, param, None)
    }

    /// Loads a half-turns expression as a [`LoadedParameter`] in the hugr, and
    /// converts it to the required parameter type.
    ///
    /// See [`PytketDecoderContext::load_half_turns`] for more details.
    pub fn load_half_turns_with_type(
        &mut self,
        param: &str,
        typ: ParameterType,
    ) -> LoadedParameter {
        self.wire_tracker
            .load_half_turns_parameter(&mut self.builder, param, Some(typ))
            .with_type(typ, &mut self.builder)
    }

    /// Returns the configuration used by the decoder.
    pub fn config(&self) -> &Arc<PytketDecoderConfig> {
        self.options.get_config()
    }

    /// Returns the options used by the decoder.
    pub fn options(&self) -> &DecodeOptions {
        &self.options
    }
}

/// Result of trying to decode pytket operation into a HUGR definition.
///
/// This flag indicates that either
/// - The operation was successful encoded and is now registered on the
///   [`PytketDecoderContext`]
/// - The operation cannot be encoded, and the context was left unchanged.
///
/// The latter is a recoverable error, as it promises that the context wasn't
/// modified. For non-recoverable errors, see [`PytketDecodeError`].
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, derive_more::Display)]
pub enum DecodeStatus {
    /// The pytket operation was successfully encoded and registered in the
    /// context.
    Success,
    /// The pytket operation could not be encoded, and the context was left
    /// unchanged.
    Unsupported,
}

/// Helper to continue exhausting the iterators in [`PytketDecoderContext::register_node_outputs`] until we have the total number of elements to report.
fn make_unexpected_node_out_error<'ty>(
    config: &PytketDecoderConfig,
    port_types: impl IntoIterator<Item = (OutgoingPort, &'ty Type)>,
    mut partial_count: RegisterCount,
    expected_qubits: usize,
    expected_bits: usize,
) -> PytketDecodeError {
    for (_, ty) in port_types {
        partial_count += config.type_to_pytket(ty).unwrap_or_default();
    }
    PytketDecodeErrorInner::UnexpectedNodeOutput {
        expected_qubits,
        expected_bits,
        circ_qubits: partial_count.qubits,
        circ_bits: partial_count.bits,
    }
    .wrap()
}
