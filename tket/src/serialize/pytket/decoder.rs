//! Intermediate structure for decoding [`SerialCircuit`]s into [`Hugr`]s.

mod param;
mod tracked_elem;
mod wires;

pub use param::{LoadedParameter, LoadedParameterType};
pub use tracked_elem::{TrackedBit, TrackedQubit};
pub use wires::{TrackedWires, WireData, WireTracker};

use std::collections::HashMap;
use std::sync::Arc;

use hugr::builder::{BuildHandle, Container, Dataflow, DataflowSubContainer, FunctionBuilder};
use hugr::extension::prelude::{bool_t, qb_t};
use hugr::ops::handle::{DataflowOpID, FuncID, NodeHandle};
use hugr::ops::{OpParent, OpTrait, OpType};
use hugr::std_extensions::arithmetic::float_types::float64_type;
use hugr::types::{Signature, Type, TypeRow};
use hugr::{Hugr, HugrView, Node, OutgoingPort, Wire};
use tracked_elem::{TrackedBitId, TrackedQubitId};

use itertools::Itertools;
use serde_json::json;
use tket_json_rs::circuit_json;
use tket_json_rs::circuit_json::SerialCircuit;
use tket_json_rs::register;

use super::{
    PytketDecodeError, METADATA_B_OUTPUT_REGISTERS, METADATA_B_REGISTERS,
    METADATA_INPUT_PARAMETERS, METADATA_PHASE, METADATA_Q_OUTPUT_REGISTERS, METADATA_Q_REGISTERS,
};
use crate::extension::rotation::rotation_type;
use crate::serialize::pytket::config::PytketDecoderConfig;
use crate::serialize::pytket::extension::{build_opaque_tket_op, RegisterCount};
use crate::serialize::pytket::PytketDecodeErrorInner;

/// State of the tket circuit being decoded.
///
/// The state of an in-progress [`FunctionBuilder`] being built from a [`SerialCircuit`].
#[derive(Debug)]
pub struct PytketDecoderContext<'h> {
    /// The Hugr being built.
    pub builder: FunctionBuilder<&'h mut Hugr>,
    /// A tracker keeping track of the generated wires and their corresponding types.
    pub wire_tracker: WireTracker,
    /// Configuration for decoding commands.
    ///
    /// Contains custom operation decoders, that define translation of legacy tket
    /// commands into HUGR operations.
    config: Arc<PytketDecoderConfig>,
}

impl<'h> PytketDecoderContext<'h> {
    /// Initialize a new [`PytketDecoderContext`], using the metadata from a
    /// [`SerialCircuit`].
    ///
    /// The circuit will be defined as a new function in the HUGR.
    ///
    /// ### Parameters
    ///
    /// - `serialcirc`: The serialised circuit to decode.
    /// - `hugr`: The [`Hugr`] to define the new function in.
    /// - `fn_name`: The name of the function to create. If `None`, we will use
    ///   the name of the circuit, or "main" if the circuit has no name.
    /// - `input_types`: The types of the input wires.
    ///   If `None`, we will use qubits and bools.
    ///   Float and rotation inputs will be associated with the `input_params` names if possible.
    /// - `output_types`: The types of the output wires.
    ///   If `None`, we will use qubits and bools.
    /// - `input_params`: A list of parameter names to add to the function input.
    ///   If additional parameters are found in the circuit, they will be added after these.
    /// - `config`: The configuration for the decoder, containing custom operation decoders.
    pub(super) fn new(
        serialcirc: &SerialCircuit,
        hugr: &'h mut Hugr,
        fn_name: Option<String>,
        input_types: Option<TypeRow>,
        output_types: Option<TypeRow>,
        input_params: impl IntoIterator<Item = String>,
        config: PytketDecoderConfig,
    ) -> Result<Self, PytketDecodeError> {
        let config = Arc::new(config);
        Self::new_arc(
            serialcirc,
            hugr,
            fn_name,
            input_types,
            output_types,
            input_params,
            config,
        )
    }

    /// Initialize a new [`PytketDecoderContext`], using the metadata from a
    /// [`SerialCircuit`].
    ///
    /// The circuit will be defined as a new function in the HUGR.
    ///
    /// ### Parameters
    ///
    /// - `serialcirc`: The serialised circuit to decode.
    /// - `hugr`: The [`Hugr`] to define the new function in.
    /// - `fn_name`: The name of the function to create. If `None`, we will use
    ///   the name of the circuit, or "main" if the circuit has no name.
    /// - `input_types`: The types of the input wires.
    ///   If `None`, we will use qubits and bools.
    ///   Float and rotation inputs will be associated with the `input_params` names if possible.
    /// - `output_types`: The types of the output wires.
    ///   If `None`, we will use qubits and bools.
    /// - `input_params`: A list of parameter names to add to the function input.
    ///   If additional parameters are found in the circuit, they will be added after these.
    /// - `config`: The configuration for the decoder, containing custom operation decoders.
    fn new_arc(
        serialcirc: &SerialCircuit,
        hugr: &'h mut Hugr,
        fn_name: Option<String>,
        input_types: Option<TypeRow>,
        output_types: Option<TypeRow>,
        input_params: impl IntoIterator<Item = String>,
        config: Arc<PytketDecoderConfig>,
    ) -> Result<Self, PytketDecodeError> {
        let num_qubits = serialcirc.qubits.len();
        let num_bits = serialcirc.bits.len();
        let input_types = input_types.unwrap_or_else(|| {
            [vec![qb_t(); num_qubits], vec![bool_t(); num_bits]]
                .concat()
                .into()
        });
        let output_types = output_types.unwrap_or_else(|| {
            [vec![qb_t(); num_qubits], vec![bool_t(); num_bits]]
                .concat()
                .into()
        });
        let sig = Signature::new(input_types.clone(), output_types.clone());

        let name = match (fn_name, &serialcirc.name) {
            (Some(name), _) => name,
            (None, Some(serial_name)) => serial_name.clone(),
            (None, None) => "".to_string(),
        };
        let mut dfg: FunctionBuilder<&mut Hugr> =
            FunctionBuilder::with_hugr(hugr, name, sig).unwrap();

        Self::init_metadata(&mut dfg, serialcirc);
        let wire_tracker =
            Self::init_wire_tracker(serialcirc, &mut dfg, &input_types, input_params, &config)?;

        if !serialcirc.phase.is_empty() {
            // TODO - add a phase gate
            // <https://github.com/CQCL/tket2/issues/598>
            // let phase = Param::new(serialcirc.phase);
            // decoder.add_phase(phase);
        }

        Ok(PytketDecoderContext {
            builder: dfg,
            wire_tracker,
            config,
        })
    }

    /// Store the serialised circuit information as HUGR metadata,
    /// so it can be reused later when re-encoding the circuit.
    fn init_metadata(dfg: &mut FunctionBuilder<&mut Hugr>, serialcirc: &SerialCircuit) {
        // Metadata. The circuit requires "name", and we store other things that
        // should pass through the serialization roundtrip.
        dfg.set_metadata(METADATA_PHASE, json!(serialcirc.phase));
        dfg.set_metadata(METADATA_Q_REGISTERS, json!(serialcirc.qubits));
        dfg.set_metadata(METADATA_B_REGISTERS, json!(serialcirc.bits));

        // Compute the output register reordering, and store it in the metadata.
        //
        // The `implicit_permutation` field is a dictionary mapping input
        // registers to output registers on the same path.
        //
        // Here we store an ordered list showing the order in which the input
        // registers appear in the output.
        //
        // For a circuit with three qubit registers 0, 1, 2 and an implicit
        // permutation {0 -> 1, 1 -> 2, 2 -> 0}, `output_to_input` will be
        // {1 -> 0, 2 -> 1, 0 -> 2} and the output order will be [2, 0, 1].
        // That is, at position 0 of the output we'll see the register originally
        // named 2, at position 1 the register originally named 0, and so on.
        let mut output_qubits = Vec::with_capacity(serialcirc.qubits.len());
        let mut output_bits = Vec::with_capacity(serialcirc.bits.len());
        let output_to_input: HashMap<register::ElementId, register::ElementId> = serialcirc
            .implicit_permutation
            .iter()
            .map(|p| (p.1.clone().id, p.0.clone().id))
            .collect();
        for qubit in &serialcirc.qubits {
            // For each output position, find the input register that should be there.
            output_qubits.push(output_to_input.get(&qubit.id).unwrap_or(&qubit.id).clone());
        }
        for bit in &serialcirc.bits {
            // For each output position, find the input register that should be there.
            output_bits.push(output_to_input.get(&bit.id).unwrap_or(&bit.id).clone());
        }
        dfg.set_metadata(METADATA_Q_OUTPUT_REGISTERS, json!(output_qubits));
        dfg.set_metadata(METADATA_B_OUTPUT_REGISTERS, json!(output_bits));
    }

    /// Initialize the wire tracker with the input wires.
    ///
    /// Utility method for [`PytketDecoderContext::new_arc`].
    fn init_wire_tracker(
        serialcirc: &SerialCircuit,
        dfg: &mut FunctionBuilder<&mut Hugr>,
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
            .map(|qb| TrackedQubit::new(Arc::new(qb.id.clone())));
        let mut bits = serialcirc
            .bits
            .iter()
            .map(|bit| TrackedBit::new(Arc::new(bit.id.clone())));
        let mut signature_reg_count = RegisterCount::default();
        for (wire, ty) in dfg.input_wires().zip(input_types.iter()) {
            let elem_counts = config.type_to_pytket(ty).unwrap_or_default();
            signature_reg_count += elem_counts;

            if elem_counts.is_empty() {
                // Input is ignored.
                continue;
            }

            let wire_qubits = qubits.by_ref().take(elem_counts.qubits);
            let wire_bits = bits.by_ref().take(elem_counts.bits);
            wire_tracker.track_wire(wire, Arc::new(ty.clone()), wire_qubits, wire_bits)?;

            if elem_counts.params > 0 {
                if elem_counts != RegisterCount::only_params(1) {
                    return Err(PytketDecodeErrorInner::UnsupportedParametersInInput {
                        ty: ty.to_string(),
                        num_params: elem_counts.params,
                    }
                    .wrap());
                }
                let Some(param) = input_params.next() else {
                    return Err(PytketDecodeErrorInner::MissingParametersInInput {
                        num_params_given: signature_reg_count.params - 1,
                    }
                    .wrap());
                };
                wire_tracker.register_input_parameter(wire, param)?;
            }
        }

        // Insert any remaining parameters as new inputs
        for param in input_params {
            let wire = dfg.add_input(rotation_type());
            wire_tracker.register_input_parameter(wire, param)?;
        }

        if qubits.next().is_some() || bits.next().is_some() {
            return Err(PytketDecodeErrorInner::InvalidInputSignature {
                input_types: input_types.iter().map(|t| t.to_string()).collect(),
                expected_qubits: num_qubits,
                expected_bits: num_bits,
                circ_qubits: serialcirc.qubits.len(),
                circ_bits: serialcirc.bits.len(),
            }
            .wrap());
        }

        Ok(wire_tracker)
    }

    /// Finish building the function definition for the legacy tket circuit.
    ///
    /// After this call, the HUGR passed to the [`PytketDecoderContext::new`]
    /// constructor will contain the fully defined function.
    ///
    /// The original Hugr entrypoint is _not_ modified, it must be set by the
    /// caller if required.
    pub(super) fn finish(mut self) -> Result<BuildHandle<FuncID<true>>, PytketDecodeError> {
        // Order the final wires according to the serial circuit register order.
        let qubits = self
            .wire_tracker
            .known_pytket_qubits()
            .map(|elem| elem.pytket_register_arc())
            .collect_vec();
        let bits = self
            .wire_tracker
            .known_pytket_bits()
            .map(|elem| elem.pytket_register_arc())
            .collect_vec();
        let function_type = self
            .builder
            .hugr()
            .get_optype(self.builder.container_node())
            .inner_function_type()
            .unwrap();
        let expected_output_types = function_type.output_types().iter().cloned().collect_vec();
        let output_wires = self
            .wire_tracker
            .wire_inputs_for_command(
                &mut self.builder,
                qubits.iter().map(|r| r.as_ref()),
                bits.iter().map(|r| r.as_ref()),
                Vec::new(),
            )
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

        self.builder
            .finish_with_outputs(output_wires)
            .map_err(PytketDecodeError::custom)
    }

    /// Decode a list of pytket commands.
    pub(super) fn run_decoder(
        &mut self,
        commands: Vec<circuit_json::Command>,
    ) -> Result<(), PytketDecodeError> {
        let config = self.config.clone();
        for com in commands {
            self.process_command(com, config.as_ref())?;
        }
        Ok(())
    }

    /// Add a tket1 [`circuit_json::Command`] from the serial circuit to the
    /// decoder.
    pub(super) fn process_command(
        &mut self,
        command: circuit_json::Command,
        config: &PytketDecoderConfig,
    ) -> Result<(), PytketDecodeError> {
        let op_type = command.op.op_type.clone();
        let circuit_json::Command { op, args, opgroup } = command;

        let num_qubits = args
            .iter()
            .find_position(|arg| self.wire_tracker.is_known_bit(arg))
            .map_or(args.len(), |(i, _)| i);
        let (qubits, bits) = args.split_at(num_qubits);

        let input_wires = self
            .wire_tracker
            .wire_inputs_for_command(
                &mut self.builder,
                qubits,
                bits,
                op.params.iter().flat_map(|v| v.iter().map(|p| p.as_str())),
            )
            .map_err(|e| e.pytket_op(&op_type))?;
        match config.op_to_hugr(&op, &input_wires, &opgroup, self)? {
            DecodeStatus::Success => {}
            DecodeStatus::Unsupported => {
                // The command couldn't be translated into a native HUGR counterpart, so
                // we generate an opaque `Tk1Op` instead.
                build_opaque_tket_op(op, &input_wires, &opgroup, self)?;
            }
        }
        Ok(())
    }
}

/// Public API, used by the [`PytketDecoder`][super::extension::PytketDecoder] implementers.
impl<'h> PytketDecoderContext<'h> {
    /// Add a new node to the HUGR with with bare qubit, bit, and parameter
    /// wires.
    ///
    /// Inserts the new node into the HUGR, connects its input ports and
    /// registers the node's output wires in the wire tracker.
    ///
    /// The qubits registers in `wires` are reused between the operation inputs
    /// and outputs. Bit registers, on the other hand, are not reused. We use
    /// the first registers in `wires` for the bit inputs and the remaining
    /// registers for the outputs.
    ///
    /// # Parameters
    ///
    /// - `op`: The operation to add.
    /// - `wires`: The wire set to use as input and outputs.
    ///
    /// # Errors
    ///
    /// - Returns an error if the optype signature contains any complex type.
    /// - Returns an error if the input wire set cannot be mapped to the node's
    ///   input ports.
    /// - Returns an error if the node's output ports cannot be assigned to
    ///   arguments from the input wire set.
    pub fn add_node_with_bare_wires(
        &mut self,
        op: impl Into<OpType>,
        wires: &TrackedWires,
    ) -> Result<BuildHandle<DataflowOpID>, PytketDecodeError> {
        let op: OpType = op.into();

        let Some(sig) = op.dataflow_signature() else {
            return Err(PytketDecodeError::custom(format!(
                "Cannot decode non-dataflow operation {op}"
            )));
        };

        /// Check that the type iterator only contains bare qubit, bit, and parameter types.
        ///
        /// Returns `true` if the types are all bare, `false` otherwise.
        fn check_bare_types<'ty>(tys: impl Iterator<Item = &'ty Type>) -> bool {
            for ty in tys {
                if ty != &qb_t()
                    && ty != &bool_t()
                    && ty != &rotation_type()
                    && ty != &float64_type()
                {
                    return false;
                }
            }
            true
        }

        // Check that the operation only uses bare types in its signature.
        if !check_bare_types(sig.output_types().iter()) {
            return Err(PytketDecodeError::custom(format!(
                "Cannot decode operation {op} with complex signature: {sig}"
            )));
        };
        if !check_bare_types(sig.input_types().iter()) {
            return Err(PytketDecodeError::custom(format!(
                "Cannot decode operation {op} with complex signature: {sig}"
            )));
        };

        // Compute the amount of elements required by the operation,
        // and the amount of elements in the input wires.
        //
        // Qubit registers get reused for both input and output, bit registers are not.
        let op_input_count: RegisterCount = sig
            .input_types()
            .iter()
            .map(|ty| self.config.type_to_pytket(ty).unwrap_or_default())
            .sum();
        let op_output_count: RegisterCount = sig
            .output_types()
            .iter()
            .map(|ty| self.config.type_to_pytket(ty).unwrap_or_default())
            .sum();
        if op_output_count.params > 0 {
            return Err(PytketDecodeError::custom(format!(
                "Tried to decode a Pytket op into a HUGR {op} with output parameters. Signature: {sig}"
            )));
        }
        let op_reg_count = RegisterCount::new(
            op_input_count.qubits.max(op_output_count.qubits),
            op_input_count.bits + op_output_count.bits,
            op_input_count.params,
        );
        let wire_reg_count = wires.register_count();

        // Check that the input wires have the amount of elements required by the operation.
        if !check_bare_types(wires.wire_types()) || wire_reg_count != op_reg_count {
            let expected_types = sig
                .input_types()
                .iter()
                .map(ToString::to_string)
                .collect_vec();
            let actual_types = wires.wire_types().map(ToString::to_string).collect_vec();
            return Err(PytketDecodeErrorInner::UnexpectedInputWires {
                expected_values: op_reg_count.qubits + op_reg_count.bits,
                expected_params: op_reg_count.params,
                actual_values: wire_reg_count.qubits + wire_reg_count.bits,
                actual_params: wire_reg_count.params,
                expected_types: Some(expected_types),
                actual_types: Some(actual_types),
            }
            .wrap()
            .hugr_op(&op));
        };

        // Gather the input wires and output wires to the operation.
        //
        // Qubit registers get reused for both input and output, bit registers are not.
        let mut bits = wires.iter_values().filter(|w| w.num_bits() == 1);
        let input_wires =
            {
                let mut input_wires = Vec::with_capacity(sig.input_count());
                let mut qubits = wires.iter_values().filter(|w| w.num_qubits() == 1);
                let mut params = wires.iter_parameters();
                for ty in sig.input_types() {
                    match ty {
                        ty if ty == &qb_t() => input_wires.push(qubits.next().unwrap().wire()),
                        ty if ty == &bool_t() => input_wires.push(bits.next().unwrap().wire()),
                        // For the parameter inputs, we add float/rotation conversion operations as needed.
                        ty if ty == &rotation_type() => input_wires
                            .push(params.next().unwrap().as_rotation(&mut self.builder).wire),
                        ty if ty == &float64_type() => input_wires
                            .push(params.next().unwrap().as_float(&mut self.builder).wire),
                        _ => unreachable!("Unsupported type"),
                    }
                }
                input_wires
            };
        let (output_qubits, output_bits) = {
            let mut output_qubits = Vec::with_capacity(op_output_count.qubits);
            let mut output_bits = Vec::with_capacity(op_output_count.bits);
            let mut qubits = wires.qubits(self);
            for ty in sig.output_types() {
                match ty {
                    ty if ty == &qb_t() => output_qubits.push(qubits.next().unwrap()),
                    ty if ty == &bool_t() => {
                        output_bits.push(bits.next().unwrap().bits(self).next().unwrap())
                    }
                    _ => unreachable!("Pytket ops cannot have parameter outputs"),
                }
            }
            (output_qubits, output_bits)
        };

        let node = self
            .builder
            .add_dataflow_op(op, input_wires)
            .map_err(PytketDecodeError::custom)?;

        self.register_node_outputs(node.node(), output_qubits, output_bits)?;

        Ok(node)
    }

    /// Given a new node in the HUGR, register all of its output wires in the
    /// tracker.
    ///
    /// Consumes the bits and qubits in order, and returns an error if
    /// they don't match the element count in the wires exactly.
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
            let counts = self.config.type_to_pytket(ty).unwrap_or_default();
            reg_count += counts;

            // Get the qubits and bits for this wire.
            let wire_qubits = qubits.by_ref().take(counts.qubits).collect_vec();
            let wire_bits = bits.by_ref().take(counts.bits).collect_vec();
            if wire_qubits.len() != counts.qubits || wire_bits.len() != counts.bits {
                let expected_qubits = reg_count.qubits - counts.qubits + wire_qubits.len();
                let expected_bits = reg_count.bits - counts.bits + wire_bits.len();
                return Err(make_unexpected_node_out_error(
                    &self.config,
                    port_types,
                    reg_count,
                    expected_qubits,
                    expected_bits,
                ));
            }

            self.wire_tracker
                .track_wire(wire, Arc::new(ty.clone()), wire_qubits, wire_bits)?;
        }

        Ok(())
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
