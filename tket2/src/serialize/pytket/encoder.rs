//! Intermediate structure for encoding [`Circuit`]s into [`SerialCircuit`]s.

mod config;
mod unit_generator;
mod unsupported_tracker;
mod value_tracker;

pub use config::{default_encoder_config, Tk1Encoder, Tk1EncoderConfig};
use hugr::ops::{NamedOp, OpTrait, OpType};
use hugr::types::EdgeKind;

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use hugr::{Direction, HugrView, Wire};
use itertools::Itertools;
use portgraph::algorithms::TopoSort;
use portgraph::LinkView;
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

/// The state of an in-progress [`SerialCircuit`] being built from a [`Circuit`].
#[derive(derive_more::Debug)]
#[debug(bounds(H: HugrView))]
pub struct Tk1EncoderContext<H: HugrView> {
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
    config: Arc<Tk1EncoderConfig<H>>,
}

impl<H: HugrView> Tk1EncoderContext<H> {
    /// Create a new [`JsonEncoder`] from a [`Circuit`].
    pub(super) fn new(
        circ: &Circuit<H>,
        config: Tk1EncoderConfig<H>,
    ) -> Result<Self, Tk1ConvertError<H::Node>> {
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
            config: Arc::new(config),
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
        let portgraph = hugr.portgraph();
        let hierarchy = hugr.hierarchy();
        let region =
            portgraph::view::FlatRegion::new(&portgraph, &hierarchy, hugr.get_pg_index(root));

        // Collect al initial nodes in the region (nodes with no predecessors).
        // Also compute a map from portgraph indices into `H::Node`s.
        //
        // TODO: Ideally we'd have a `hugr.from_pg_index` method, and avoid computing the map.
        // We should revisit this once the `base_hugr` refactor is done.
        // <https://github.com/CQCL/hugr/issues/1926>
        let mut initials = Vec::new();
        let mut from_pg_index = HashMap::new();
        for node in hugr.children(root) {
            let pg_node = hugr.get_pg_index(node);
            from_pg_index.insert(pg_node, node);
            if region.input_links(pg_node).next().is_none() {
                initials.push(pg_node);
            }
        }

        let topo: TopoSort<'_, _, HashSet<_>> =
            portgraph::algorithms::toposort(region, initials, portgraph::Direction::Outgoing);
        for pg_node in topo {
            let node = from_pg_index[&pg_node];
            self.try_encode_node(node, circ)?;
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

    /// Given a node in the HUGR, returns all the [`TrackedValue`]s associated
    /// with its inputs.
    ///
    /// These values can be used with the [`Tk1EncoderContext::values`] tracker
    /// to retrieve the corresponding pytket registers and parameter
    /// expressions. See [`ValueTracker::qubit_register`],
    /// [`ValueTracker::bit_register`], and [`ValueTracker::param_expression`].
    pub fn get_input_values(
        &mut self,
        node: H::Node,
        circ: &Circuit<H>,
    ) -> Result<(Vec<TrackedQubit>, Vec<TrackedBit>, Vec<TrackedParam>), Tk1ConvertError<H::Node>>
    {
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
        Ok((qubits, bits, params))
    }

    /// Helper to emit a new tket1 command corresponding to a single HUGR node.
    ///
    /// This call will fail if the node has parameter outputs. Use
    /// [`Tk1EncoderContext::emit_node_with_out_params`] instead.
    ///
    /// See [`Tk1EncoderContext::emit_command`] for more general cases.
    ///
    /// ## Arguments
    ///
    /// - `tk1_operation`: The tket1 operation type to emit.
    /// - `node`: The HUGR for which to emit the command. Qubits and bits are
    ///   automatically retrieved from the node's inputs/outputs.
    /// - `circ`: The circuit containing the node.
    pub fn emit_node(
        &mut self,
        tk1_optype: tket_json_rs::OpType,
        node: H::Node,
        circ: &Circuit<H>,
    ) -> Result<(), Tk1ConvertError<H::Node>> {
        self.emit_node_with_out_params(tk1_optype, node, circ, |_, _| None)
    }

    /// Helper to emit a new tket1 command corresponding to a single HUGR node,
    /// with parameter outputs. Use [`Tk1EncoderContext::emit_node`]
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
    pub fn emit_node_with_out_params(
        &mut self,
        tk1_optype: tket_json_rs::OpType,
        node: H::Node,
        circ: &Circuit<H>,
        param_map: impl FnMut(usize, &[String]) -> Option<String>,
    ) -> Result<(), Tk1ConvertError<H::Node>> {
        self.emit_node_command(
            node,
            circ,
            param_map,
            move |qubit_count, bit_count, params| {
                make_tk1_operation(tk1_optype, qubit_count, bit_count, params)
            },
        )
    }

    /// Helper to emit a new tket1 command corresponding to a single HUGR node,
    /// using a custom operation generator and computing output parameter
    /// expressions. Use [`Tk1EncoderContext::emit_node`] or
    /// [`Tk1EncoderContext::emit_node_with_out_params`] when pytket operation
    /// can be defined directly from a [`tket_json_rs::OpType`].
    ///
    /// See [`Tk1EncoderContext::emit_command`] for a general case emitter.
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
    /// - `make_operation`: A function that takes the number of qubits, bits, and
    ///   the list of input parameter expressions and returns a pytket operation.
    ///   See [`make_tk1_operation`] for a helper function to create it.
    pub fn emit_node_command(
        &mut self,
        node: H::Node,
        circ: &Circuit<H>,
        mut param_map: impl FnMut(usize, &[String]) -> Option<String>,
        make_operation: impl FnOnce(usize, usize, &[String]) -> tket_json_rs::circuit_json::Operation,
    ) -> Result<(), Tk1ConvertError<H::Node>> {
        let (qubits, bits, params) = self.get_input_values(node, circ)?;
        let params: Vec<String> = params
            .into_iter()
            .map(|p| self.values.param_expression(p).to_owned())
            .collect();

        // Update the values in the node's outputs.
        //
        // We preserve the order of linear values in the input
        let optype = circ.hugr().get_optype(node);
        let sig = optype.dataflow_signature();
        let other_output_port = optype.other_output_port();
        let mut qubit_iterator = qubits.iter();
        let mut param_counter = 0;
        for output in circ.hugr().node_outputs(node) {
            // Ignore order edges.
            if Some(output) == other_output_port {
                continue;
            }
            let type_counts: RegisterCount =
                match sig.as_ref().and_then(|s| s.out_port_type(output)) {
                    Some(ty) => self.config.type_to_pytket(ty)?,
                    None => {
                        // It must be a static type.
                        debug_assert_eq!(
                            Some(output),
                            optype.static_output_port(),
                            "Failed to get output type of port {output} in op {}",
                            optype.name()
                        );
                        match optype.static_port_kind(Direction::Outgoing).unwrap() {
                            EdgeKind::Const(ty) => self.config.type_to_pytket(&ty)?,
                            // Unsupported wire type, no pytket types to track.
                            _ => None,
                        }
                    }
                }
                .unwrap_or_default();

            // Collect the values associated with the output wire.
            let out_wire = Wire::new(node, output);
            let mut out_wire_values = Vec::with_capacity(type_counts.total());
            for _ in 0..type_counts.qubits {
                let qb = match qubit_iterator.next() {
                    Some(qb) => *qb,
                    None => {
                        // If we already have a matching output for all input qubits,
                        // generate a fresh ID.
                        self.values.new_qubit()
                    }
                };
                out_wire_values.push(TrackedValue::Qubit(qb));
            }
            for _ in 0..type_counts.bits {
                let b = self.values.new_bit();
                out_wire_values.push(TrackedValue::Bit(b));
            }
            for _ in 0..type_counts.params {
                if let Some(out_param_expression) = param_map(param_counter, &params) {
                    let p = self.values.new_param(out_param_expression);
                    out_wire_values.push(TrackedValue::Param(p));
                }
                param_counter += 1;
            }
            self.values
                .register_values(out_wire, out_wire_values, circ)?;
        }

        // Preserve the pytket opgroup, if it got stored in the metadata.
        let opgroup: Option<String> = circ
            .hugr()
            .get_metadata(node, METADATA_OPGROUP)
            .and_then(serde_json::Value::as_str)
            .map(ToString::to_string);

        let op = make_operation(qubits.len(), bits.len(), &params);
        self.emit_command(op, &qubits, &bits, opgroup);
        Ok(())
    }

    /// Emit a new tket1 command.
    ///
    /// This is a general-purpose command that can be used to emit any tket1
    /// operation, not necessarily corresponding to a specific HUGR node.
    ///
    /// In general you should prefer using [`Tk1EncoderContext::emit_node`] or
    /// [`Tk1EncoderContext::emit_node_with_out_params`] as they automatically
    /// compute the input qubits and bits from the HUGR node, and ensure that
    /// output wires get their new values registered on the
    /// [`Tk1EncoderContext::values`] tracker.
    ///
    /// ## Arguments
    ///
    /// - `tk1_operation`: The tket1 operation to emit. See
    ///   [`make_tk1_operation`] for a helper function to create it.
    /// - `qubits`: The qubit registers to use as inputs/outputs of the pytket op.
    ///   Normally obtained from a HUGR node's inputs using [`Tk1EncoderContext::get_input_values`]
    ///   or allocated via [`ValueTracker::new_qubit`].
    /// - `bits`: The bit registers to use as inputs/outputs of the pytket op.
    ///   Normally obtained from a HUGR node's inputs using [`Tk1EncoderContext::get_input_values`]
    ///   or allocated via [`ValueTracker::new_bit`].
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
    /// of the encoders could process it and the node got added to the
    /// unsupported set.
    fn try_encode_node(
        &mut self,
        node: H::Node,
        circ: &Circuit<H>,
    ) -> Result<bool, Tk1ConvertError<H::Node>> {
        let optype = circ.hugr().get_optype(node);

        // TODO: Boxes and non-custom optypes

        // Try to encode the operation using each of the registered encoders.
        //
        // If none of the encoders can handle the operation, we just add it to
        // the unsupported tracker and move on.
        match optype {
            OpType::ExtensionOp(op) => {
                let config = Arc::clone(&self.config);
                if config.op_to_pytket(node, op, circ, self)? {
                    return Ok(true);
                }
            }
            _ => {}
        }

        self.unsupported.record_node(node, circ);
        Ok(false)
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
    params: &[String],
) -> circuit_json::Operation {
    let mut op = circuit_json::Operation::default();
    op.op_type = tk1_optype;
    op.n_qb = Some(qubit_count as u32);
    op.params = match params.is_empty() {
        false => Some(params.to_owned()),
        true => None,
    };
    op.signature = Some([vec!["Q".into(); qubit_count], vec!["B".into(); bit_count]].concat());
    op
}
