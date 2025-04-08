//! Intermediate structure for encoding [`Circuit`]s into [`SerialCircuit`]s.

mod unit_generator;
mod unsupported_tracker;
mod value_tracker;

use std::borrow::Cow;
use std::collections::HashMap;

use hugr::extension::ExtensionId;
use hugr::ops::OpType;
use hugr::types::{CustomType, Type, TypeEnum};

use hugr::extension::prelude::bool_t;
use hugr::{HugrView, Wire};
use itertools::Itertools;
use tket_json_rs::circuit_json::{self, SerialCircuit};
use unsupported_tracker::UnsupportedTracker;
use value_tracker::{
    RegisterCount, TrackedBit, TrackedParam, TrackedQubit, TrackedValue, ValueTracker,
};

use super::{
    OpConvertError, Tk1ConvertError, METADATA_B_OUTPUT_REGISTERS, METADATA_OPGROUP, METADATA_PHASE,
    METADATA_Q_OUTPUT_REGISTERS, METADATA_Q_REGISTERS,
};
use crate::circuit::Circuit;

pub fn default_encoder_config<H: HugrView>() -> Tk1EncoderConfig<H> {
    // TODO: Add prelude & tket2 encoders
    Tk1EncoderConfig::new()
}

/// An encoder of HUGR operations and types that transform them
/// into pytket primitives.
pub trait Tk1Encoder<H: HugrView> {
    /// The name of the extension this encoder/decoder is for.
    ///
    /// [`Tk1Encoder::op_to_pytket`] and [`Tk1Encoder::type_to_pytket`] will
    /// only be called for operations/types of these extensions.
    fn extensions(&self) -> Vec<Cow<'_, ExtensionId>>;

    /// Given a node in the HUGR circuit and its operation type, try to convert
    /// it to a pytket operation and add it to the pytket encoder.
    ///
    /// Returns `true` if the operation was successfully converted. If that is
    /// the case, no further encoders will be called.
    ///
    /// If the operation is not supported by the encoder, return `false`.
    fn op_to_pytket(
        &self,
        node: H::Node,
        op: &OpType,
        circ: &Circuit<H>,
        encoder: &mut Tk1EncoderContext<H>,
    ) -> Result<bool, Tk1ConvertError<H::Node>>;

    /// Given a HUGR type, return the number of qubits, bits, and sympy
    /// parameters of its pytket counterpart.
    ///
    /// If the type is not supported by the encoder, return `None`.
    fn type_to_pytket(
        &self,
        #[allow(unused)] op: &CustomType,
    ) -> Result<Option<RegisterCount>, Tk1ConvertError<H::Node>> {
        Ok(None)
    }
}

/// Configuration for converting [`Circuit`] into [`SerialCircuit`].
///
/// Contains custom encoders that define translations for HUGR operations and types
/// into pytket primitives.
#[derive(derive_more::Debug)]
#[debug(bounds(H: HugrView))]
struct Tk1EncoderConfig<H: HugrView> {
    /// Operation encoders
    #[debug(skip)]
    encoders: Vec<Box<dyn Tk1Encoder<H>>>,
    /// Pre-computed map from extension ids to corresponding encoders in
    /// `encoders`, identified by their index.
    #[debug("{:?}", extension_encoders.keys().collect_vec())]
    extension_encoders: HashMap<ExtensionId, Vec<usize>>,
}

impl<H: HugrView> Tk1EncoderConfig<H> {
    /// Create a new [`Tk1EncoderConfig`] with no encoders.
    pub fn new() -> Self {
        Self {
            encoders: vec![],
            extension_encoders: HashMap::new(),
        }
    }

    /// Add an encoder to the configuration.
    pub fn add_encoder(&mut self, encoder: impl Tk1Encoder<H> + 'static) {
        let idx = self.encoders.len();

        for ext in encoder.extensions() {
            self.extension_encoders
                .entry(ext.into_owned())
                .or_default()
                .push(idx);
        }
        self.encoders.push(Box::new(encoder));
    }

    /// List the extensions supported by the encoders.
    ///
    /// Use [`Tk1EncoderConfig::add_encoder`] to extend this list.
    pub fn supported_extensions(&self) -> impl Iterator<Item = &ExtensionId> {
        self.extension_encoders.keys()
    }

    /// Translate a HUGR type into a count of qubits, bits, and parameters,
    /// using the registered custom encodes.
    ///
    /// Only tuple sums, bools, and custom types are supported.
    /// Other types will return `None`.
    pub fn type_to_pytket(&self, typ: &Type) -> Result<Option<RegisterCount>, OpConvertError> {
        match typ.as_type_enum() {
            TypeEnum::Sum(sum) => {
                if sum == bool_t() {
                    return Ok(Some(RegisterCount {
                        qubits: 0,
                        bits: 1,
                        params: 0,
                    }));
                }
                if let Some(tuple) = sum.as_tuple() {
                    let count: Result<Option<RegisterCount>, OpConvertError> = tuple
                        .iter()
                        .map(|ty| {
                            match ty.try_into() {
                                Ok(ty) => Ok(self.type_to_pytket(ty)?),
                                // Sum types with row variables (variable tuple lengths) are not supported.
                                Err(_) => Ok(None),
                            }
                        })
                        .sum();
                    return count;
                }
            }
            TypeEnum::Extension(custom) => {
                let type_ext = custom.extension();
                for encoder in self.encoders_for_extension(type_ext) {
                    if let Some(count) = encoder.type_to_pytket(custom)? {
                        return Ok(Some(count));
                    }
                }
            }
            _ => {}
        }
        Ok(None)
    }

    /// Lists the encoders that can handle a given extension.
    fn encoders_for_extension(
        &self,
        ext: &ExtensionId,
    ) -> impl Iterator<Item = &Box<dyn Tk1Encoder<H>>> {
        self.extension_encoders
            .get(ext)
            .into_iter()
            .flat_map(move |idxs| idxs.iter().map(move |idx| &self.encoders[*idx]))
    }
}

/// The state of an in-progress [`SerialCircuit`] being built from a [`Circuit`].
#[derive(derive_more::Debug)]
#[debug(bounds(H: HugrView))]
pub(super) struct Tk1EncoderContext<H: HugrView> {
    /// The name of the circuit being encoded.
    name: Option<String>,
    /// Global phase value.
    ///
    /// Defaults to "0" unless the circuit has a [METADATA_PHASE] metadata
    /// entry.
    phase: String,
    /// The already-encoded serialised pytket commands.
    commands: Vec<circuit_json::Command>,
    /// A tracker for qubit/bit/parameter values associated with the circuit's wires.
    ///
    /// Contains methods to update the registers in the circuit being built.
    pub values: ValueTracker<H::Node>,
    /// A tracker for unsupported regions of the circuit.
    unsupported: UnsupportedTracker<H::Node>,
    /// Configuration for the encoding.
    ///
    /// Contains custom operation/type encoders.
    config: Tk1EncoderConfig<H>,
}

impl<H: HugrView> Tk1EncoderContext<H> {
    /// Create a new [`JsonEncoder`] from a [`Circuit`].
    pub(super) fn new(
        circ: &Circuit<H>,
        config: Tk1EncoderConfig<H>,
    ) -> Result<Self, Tk1ConvertError> {
        let name = circ.name().map(str::to_string);
        let hugr = circ.hugr();

        // Recover other parameters stored in the metadata
        let phase = match hugr.get_metadata(circ.parent(), METADATA_PHASE) {
            Some(p) => p.as_str().unwrap().to_string(),
            None => "0".to_string(),
        };

        Ok(Self {
            name,
            phase,
            commands: vec![],
            values: ValueTracker::new(circ, &config)?,
            unsupported: UnsupportedTracker::new(circ),
            config,
        })
    }

    /// Traverse the circuit in topological order, encoding the nodes as pytket commands.
    ///
    /// Returns the final [`SerialCircuit`] if successful.
    pub(super) fn run_encoder(
        &mut self,
        circ: &Circuit<H>,
    ) -> Result<(), Tk1ConvertError<H::Node>> {
        // Normally we'd use `SiblingGraph` here, but it doesn't support generic node types.
        // See https://github.com/CQCL/hugr/issues/2010
        let hugr = circ.hugr();
        let root = circ.parent();
        let region = portgraph::view::FlatRegion::new(
            &hugr.portgraph(),
            &hugr.hierarchy(),
            hugr.get_pg_index(root),
        );
        let mut nodes = petgraph::visit::Topo::new(&region);
        while let Some(node) = nodes.next(&region) {
            // Try to encode the single node as pytket commands.
            // If it cannot be encoded, track it as part of an unsupported region.
            if !self.try_encode_node(node, circ)? {
                self.unsupported.record_node(node, circ);
            }
        }
        Ok(())
    }

    /// Finish building and return the final [`SerialCircuit`].
    pub(super) fn finish(
        self,
        circ: &Circuit<H>,
    ) -> Result<SerialCircuit, Tk1ConvertError<H::Node>> {
        let mut final_values = self.values.finish(circ)?;

        let mut implicit_permutation = final_values.qubit_permutation;
        implicit_permutation.append(&mut final_values.bit_permutation);

        let mut ser = SerialCircuit::new(self.name, self.phase);

        ser.commands = self.commands;
        ser.qubits = final_values.qubits.into_iter().map_into().collect();
        ser.bits = final_values.bits.into_iter().map_into().collect();
        ser.implicit_permutation = implicit_permutation;
        ser.number_of_ws = None;
        Ok(ser)
    }

    /// Helper to emit a new tket1 command corresponding to a single HUGR node.
    ///
    /// This call will fail if the node has parameter outputs. Use
    /// [`Tk1EncoderContext::emit_command_for_node_with_params`] instead.
    ///
    /// See [`Tk1EncoderContext::emit_command`] for more general cases.
    ///
    /// ## Arguments
    ///
    /// - `tk1_operation`: The tket1 operation type to emit.
    /// - `node`: The HUGR for which to emit the command. Qubits and bits are
    ///   automatically retrieved from the node's inputs/outputs.
    /// - `circ`: The circuit containing the node.
    pub fn emit_command_for_node(
        &mut self,
        tk1_optype: tket_json_rs::OpType,
        node: H::Node,
        circ: &Circuit<H>,
    ) -> Result<(), Tk1ConvertError<H::Node>> {
        self.emit_command_for_node_with_params(tk1_optype, node, circ, |_, _| None)
    }

    /// Helper to emit a new tket1 command corresponding to a single HUGR node,
    /// with parameter outputs. Use [`Tk1EncoderContext::emit_command_for_node`]
    /// for nodes that don't require computing parameter outputs.
    ///
    /// See [`Tk1EncoderContext::emit_command`] for more general cases.
    ///
    /// ## Arguments
    ///
    /// - `tk1_operation`: The tket1 operation type to emit.
    /// - `node`: The HUGR for which to emit the command. Qubits and bits are
    ///   automatically retrieved from the node's inputs/outputs.
    /// - `circ`: The circuit containing the node.
    /// - `param_map`: A function that given a parameter index and the list of
    ///   input parameter values returns the string expression to use for the
    ///   parameter output. Returning `None` will abort the encoding.
    pub fn emit_command_for_node_with_params<'p>(
        &mut self,
        tk1_optype: tket_json_rs::OpType,
        node: H::Node,
        circ: &Circuit<H>,
        param_map: impl FnMut(usize, &'p [&'p str]) -> Option<&'p str>,
    ) -> Result<(), Tk1ConvertError<H::Node>> {
        let mut qubits: Vec<TrackedQubit> = Vec::new();
        let mut bits: Vec<TrackedBit> = Vec::new();
        let mut params: Vec<TrackedParam> = Vec::new();

        let optype = circ.hugr().get_optype(node);
        let other_input_port = optype.other_input_port();
        for input in circ.hugr().node_inputs(node) {
            // Ignore order edges.
            if Some(input) == other_input_port {
                continue;
            }
            // Dataflow ports should have a single linked neighbour.
            let Some((neigh, neigh_out)) = circ.hugr().single_linked_output(node, input) else {
                return Err(
                    OpConvertError::UnsupportedOpSerialization { op: optype.clone() }.into(),
                );
            };
            let wire = Wire::new(neigh, neigh_out);
            for value in self.values.wire_values(wire)?.iter() {
                match value {
                    TrackedValue::Qubit(qb) => qubits.push(*qb),
                    TrackedValue::Bit(b) => bits.push(*b),
                    TrackedValue::Param(p) => params.push(*p),
                }
            }
        }
        let params: Vec<String> = params
            .into_iter()
            .map(|p| self.values.param_expression(p).to_owned())
            .collect();

        // Update the values in the node's outputs.
        //
        // We preserve the order of linear values in the input
        let other_output_port = optype.other_output_port();
        let mut qubit_iterator = qubits.iter();
        let mut bits_iterator = bits.iter();
        for output in circ.hugr().node_outputs(node) {
            // Ignore order edges.
            if Some(output) == other_output_port {
                continue;
            }
            let wire = Wire::new(node, output);
            // Each output of the node may consume multiple values. We
            //let output_counts = self.config.type_to_pytket();

            // TODO TODO TODO
            // Fetch the type of each output port
        }

        // Preserve the pytket opgroup, if it got stored in the metadata.
        let opgroup: Option<String> = circ
            .hugr()
            .get_metadata(node, METADATA_OPGROUP)
            .and_then(serde_json::Value::as_str)
            .map(ToString::to_string);

        let op = make_tk1_operation(tk1_optype, qubits.len(), bits.len(), params);
        self.emit_command(op, &qubits, &bits, opgroup);
        Ok(())
    }

    /// Emit a new tket1 command.
    ///
    /// When
    ///
    /// ## Arguments
    ///
    /// - `tk1_operation`: The tket1 operation to emit.
    ///   See [`make_tk1_operation`] for a helper function to create it.
    /// - `qubits`: The qubit registers to use as inputs/outputs.
    ///   See [`Tk1EncoderContext::update_qubit`].
    /// - `bits`: The bit registers to use as inputs/outputs.
    ///   See [`Tk1EncoderContext::get_bit_register`].
    /// - `opgroup`: A tket1 operation group identifier, if any.
    pub fn emit_command(
        &mut self,
        tk1_operation: circuit_json::Operation,
        qubits: &[TrackedQubit],
        bits: &[TrackedBit],
        opgroup: Option<String>,
    ) {
        let qubit_regs = qubits.iter().map(|&qb| self.values.qubit_register(qb));
        let bit_regs = bits.iter().map(|&b| self.values.bit_register(b));
        let command = circuit_json::Command {
            op: tk1_operation,
            args: qubit_regs.chain(bit_regs).cloned().collect(),
            opgroup,
        };

        self.commands.push(command);
    }

    /// Encode a single circuit node into pytket commands and update the
    /// encoder.
    ///
    /// Dispatches to the registered encoders, trying each in turn until one
    /// successfully encodes the operation.
    ///
    /// Returns `true` if the node was successfully encoded, or `false` if none
    /// of the encoders could process it.
    fn try_encode_node(
        &mut self,
        node: H::Node,
        circ: &Circuit<H>,
    ) -> Result<bool, Tk1ConvertError> {
        let optype = circ.hugr().get_optype(node);

        // Try to encode the operation using each of the registered encoders.
        //
        // If none of the encoders can handle the operation, we just add it to
        // the unsupported tracker and move on.
        for encoder in &mut self.config.encoders {
            if encoder
                .op_to_pytket(node, optype, circ.hugr(), self)
                .is_ok()
            {
                return Ok(true);
            }
        }

        /*
        // Register any output of the command that can be used as a TKET1 parameter.
        if self.parameters.record_parameters(&command, optype)? {
            // for now all ops that record parameters should be ignored (are
            // just constants)
            return Ok(());
        }

        // Special case for the QAlloc operation.
        // This does not translate to a TKET1 operation, we just start tracking a new qubit register.
        if optype == &Tk2Op::QAlloc.into() {
            let Some((CircuitUnit::Linear(unit_id), _, _)) = command.outputs().next() else {
                panic!("QAlloc should have a single qubit output.")
            };
            debug_assert!(self.qubits.get(unit_id).is_none());
            self.qubits.add_qubit_register(unit_id);
            return Ok(());
        }

        let Some(tk1op) = Tk1Op::try_from_optype(optype.clone())? else {
            // This command should be ignored.
            return Ok(());
        };

        // Get the registers and wires associated with the operation's inputs.
        let mut qubit_args = Vec::with_capacity(tk1op.qubit_inputs());
        let mut bit_args = Vec::with_capacity(tk1op.bit_inputs());
        let mut params = Vec::with_capacity(tk1op.num_params());
        for (unit, _, ty) in command.inputs() {
            if ty == qb_t() {
                let reg = self.unit_to_register(unit).unwrap_or_else(|| {
                    panic!(
                        "No register found for qubit input {unit} in node {}.",
                        command.node(),
                    )
                });
                qubit_args.push(reg);
            } else if ty == bool_t() {
                let reg = self.unit_to_register(unit).unwrap_or_else(|| {
                    panic!(
                        "No register found for bit input {unit} in node {}.",
                        command.node(),
                    )
                });
                bit_args.push(reg);
            } else if [rotation_type(), float64_type()].contains(&ty) {
                let CircuitUnit::Wire(param_wire) = unit else {
                    unreachable!("Angle types are not linear.")
                };
                params.push(param_wire);
            } else {
                return Err(OpConvertError::UnsupportedInputType {
                    typ: ty.clone(),
                    optype: optype.clone(),
                    node: command.node(),
                });
            }
        }

        for (unit, _, ty) in command.outputs() {
            if ty == qb_t() {
                // If the qubit is not already in the qubit tracker, add it as a
                // new register.
                let CircuitUnit::Linear(unit_id) = unit else {
                    panic!("Qubit types are linear.")
                };
                if self.qubits.get(unit_id).is_none() {
                    let reg = self.qubits.add_qubit_register(unit_id);
                    qubit_args.push(reg.clone());
                }
            } else if ty == bool_t() {
                // If the operation has any bit outputs, create a new one bit
                // register.
                //
                // Note that we do not reassign input registers to the new
                // output wires as we do not know if the bit value was modified
                // by the operation, and the old value may be needed later.
                //
                // This may cause register duplication for opaque operations
                // with input bits.
                let CircuitUnit::Wire(wire) = unit else {
                    panic!("Bool types are not linear.")
                };
                let reg = self.bits.add_bit_register(wire);
                bit_args.push(reg.clone());
            } else {
                return Err(OpConvertError::UnsupportedOutputType {
                    typ: ty.clone(),
                    optype: optype.clone(),
                    node: command.node(),
                })
                .into();
            }
        }

        let opgroup: Option<String> = command
            .metadata(METADATA_OPGROUP)
            .and_then(serde_json::Value::as_str)
            .map(ToString::to_string);

        // Convert the command's operator to a pytket serialized one. This will
        // return an error for operations that should have been caught by the
        // `record_parameters` branch above (in addition to other unsupported
        // ops).
        let mut serial_op: circuit_json::Operation = tk1op
            .serialised_op()
            .ok_or_else(|| OpConvertError::UnsupportedOpSerialization(optype.clone()))?;

        if !params.is_empty() {
            serial_op.params = Some(
                params
                    .into_iter()
                    .filter_map(|w| self.parameters.get(&w))
                    .cloned()
                    .collect(),
            )
        }
        // TODO: ops that contain free variables.
        // (update decoder to ignore them too, but store them in the wrapped op)

        let mut args = qubit_args;
        args.append(&mut bit_args);
        let command = circuit_json::Command {
            op: serial_op,
            args,
            opgroup,
        };
        self.commands.push(command);
        */

        Ok(true)
    }
}

/// Initialize a tket1 [Operation](circuit_json::Operation) to pass to
/// [`Tk1Encoder::emit_command`].
///
/// ## Arguments
/// - `tk1_optype`: The operation type to use.
/// - `qubit_count`: The number of qubits used by the operation.
/// - `bit_count`: The number of linear bits used by the operation.
/// - `params`: Parameters of the operation, expressed as string expressions.
///   Normally obtained from [`Tk1EncoderContext::get_parameter`].
pub fn make_tk1_operation(
    tk1_optype: tket_json_rs::OpType,
    qubit_count: usize,
    bit_count: usize,
    params: Vec<String>,
) -> circuit_json::Operation {
    let mut op = circuit_json::Operation::default();
    op.op_type = tk1_optype;
    op.n_qb = Some(qubit_count as u32);
    op.params = match params.is_empty() {
        false => Some(params),
        true => None,
    };
    op.signature = Some([vec!["Q".into(); qubit_count], vec!["B".into(); bit_count]].concat());
    op
}
