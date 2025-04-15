//! Intermediate structure for encoding [`Circuit`]s into [`SerialCircuit`]s.

mod config;
mod unit_generator;
mod unsupported_tracker;
mod value_tracker;

pub use config::{default_encoder_config, Tk1EncoderConfig};
use hugr::envelope::EnvelopeConfig;
use hugr::hugr::views::SiblingSubgraph;
use hugr::package::Package;
pub use value_tracker::{
    RegisterCount, TrackedBit, TrackedParam, TrackedQubit, TrackedValue, TrackedValues,
    ValueTracker,
};

use hugr::ops::{NamedOp, OpTrait, OpType};
use hugr::types::EdgeKind;

use std::borrow::Cow;
use std::collections::{BTreeSet, HashSet};
use std::sync::Arc;

use hugr::{Direction, HugrView, Wire};
use itertools::Itertools;
use portgraph::algorithms::TopoSort;
use portgraph::LinkView;
use tket_json_rs::circuit_json::{self, SerialCircuit};
use unsupported_tracker::UnsupportedTracker;

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
    /// Contains custom operation/type/const emitters.
    config: Arc<Tk1EncoderConfig<H>>,
}

impl<H: HugrView> Tk1EncoderContext<H> {
    /// Create a new [`Tk1EncoderContext`] from a [`Circuit`].
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

        // Collect all initial nodes in the region (nodes with no predecessors).
        let initials: Vec<_> = hugr
            .children(root)
            .filter_map(|node| {
                let pg_node = hugr.get_pg_index(node);
                region
                    .input_links(pg_node)
                    .next()
                    .is_none()
                    .then_some(pg_node)
            })
            .collect();

        let topo: TopoSort<'_, _, HashSet<_>> =
            portgraph::algorithms::toposort(region, initials, portgraph::Direction::Outgoing);
        let io_nodes = circ.io_nodes();
        // TODO: Use weighted topological sort to try and explore unsupported
        // ops first (that is, ops with no available emitter in `self.config`),
        // to ensure we group them as much as possible.
        for pg_node in topo {
            let node = hugr.get_node(pg_node);
            if io_nodes.contains(&node) {
                // I/O nodes are handled by `new` and `finish`.
                continue;
            }
            self.try_encode_node(node, circ)?;
        }
        Ok(())
    }

    /// Finish building and return the final [`SerialCircuit`].
    pub(super) fn finish(
        mut self,
        circ: &Circuit<H>,
    ) -> Result<SerialCircuit, Tk1ConvertError<H::Node>> {
        // Add any remaining unsupported nodes
        //
        // TODO: Test that unsupported subgraphs that don't affect any qubit/bit registers
        // are correctly encoded in pytket commands.
        while !self.unsupported.is_empty() {
            let node = self.unsupported.iter().next().unwrap();
            let unsupported_subgraph = self.unsupported.extract_component(node);
            self.emit_unsupported(unsupported_subgraph, circ)?;
        }

        let final_values = self.values.finish(circ)?;

        let mut ser = SerialCircuit::new(self.name, self.phase);

        ser.commands = self.commands;
        ser.qubits = final_values.qubits.into_iter().map_into().collect();
        ser.bits = final_values.bits.into_iter().map_into().collect();
        ser.implicit_permutation = final_values.qubit_permutation;
        ser.number_of_ws = None;
        Ok(ser)
    }

    /// Returns a reference to this encoder's configuration.
    pub fn config(&self) -> &Tk1EncoderConfig<H> {
        &self.config
    }

    /// Returns the values associated with a wire.
    ///
    /// Marks the port connection as explored. When all ports connected to the
    /// wire have been explored, the wire is removed from the tracker.
    ///
    /// If the input wire is the output of an unsupported node, a subgraph of
    /// unsupported nodes containing it will be emitted as a pytket barrier.
    ///
    /// This function SHOULD NOT be called before determining that the target
    /// operation is supported, as it will mark the wire as explored and
    /// potentially remove it from the tracker. To determine if a wire type is
    /// supported, use [`Tk1EncoderConfig::type_to_pytket`] on the encoder
    /// context's [`Tk1EncoderContext::config`].
    ///
    /// ### Errors
    ///
    /// - [`OpConvertError::WireHasNoValues`] if the wire is not tracked or has
    ///   a type that cannot be converted to pytket values.
    pub fn get_wire_values(
        &mut self,
        wire: Wire<H::Node>,
        circ: &Circuit<H>,
    ) -> Result<Cow<'_, [TrackedValue]>, Tk1ConvertError<H::Node>> {
        if self.values.peek_wire_values(wire).is_some() {
            return Ok(self.values.wire_values(wire).unwrap());
        }

        // If the wire values have not been registered yet, it may be because
        // the wire is the output of an unsupported node group.
        //
        // We need to emit the unsupported node here before returning the values.
        if self.unsupported.is_unsupported(wire.node()) {
            let unsupported_subgraph = self.unsupported.extract_component(wire.node());
            self.emit_unsupported(unsupported_subgraph, circ)?;
            debug_assert!(!self.unsupported.is_unsupported(wire.node()));
            return self.get_wire_values(wire, circ);
        }

        Err(OpConvertError::WireHasNoValues { wire }.into())
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
    ) -> Result<TrackedValues, Tk1ConvertError<H::Node>> {
        self.get_input_values_internal(node, circ, |_| true)
    }

    /// Auxiliary function used to collect the input values of a node.
    /// See [`Tk1EncoderContext::get_input_values`].
    ///
    /// Given a node in the HUGR, returns all the [`TrackedValue`]s associated
    /// with its inputs. Calls
    ///
    /// Includes a filter to decide which incoming wires to include.
    fn get_input_values_internal(
        &mut self,
        node: H::Node,
        circ: &Circuit<H>,
        wire_filter: impl Fn(Wire<H::Node>) -> bool,
    ) -> Result<TrackedValues, Tk1ConvertError<H::Node>> {
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
            if !wire_filter(wire) {
                continue;
            }

            for value in self.get_wire_values(wire, circ)?.iter() {
                match value {
                    TrackedValue::Qubit(qb) => qubits.push(*qb),
                    TrackedValue::Bit(b) => bits.push(*b),
                    TrackedValue::Param(p) => params.push(*p),
                }
            }
        }
        Ok(TrackedValues {
            qubits,
            bits,
            params,
        })
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
    /// - `output_param`: A function that given a parameter index and the list of
    ///   input parameter values returns the string expression to use for the
    ///   parameter output. Returning `None` will abort the encoding.
    pub fn emit_node_with_out_params(
        &mut self,
        tk1_optype: tket_json_rs::OpType,
        node: H::Node,
        circ: &Circuit<H>,
        output_param: impl FnMut(usize, &[String]) -> Option<String>,
    ) -> Result<(), Tk1ConvertError<H::Node>> {
        self.emit_node_command(
            node,
            circ,
            output_param,
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
    /// - `node`: The HUGR for which to emit the command. Qubits and bits are
    ///   automatically retrieved from the node's inputs/outputs.
    /// - `circ`: The circuit containing the node.
    /// - `output_param`: A function that given a parameter index and the list of
    ///   input parameter values returns the string expression to use for the
    ///   parameter output. Returning `None` will abort the encoding.
    /// - `make_operation`: A function that takes the number of qubits, bits, and
    ///   the list of input parameter expressions and returns a pytket operation.
    ///   See [`make_tk1_operation`] for a helper function to create it.
    pub fn emit_node_command(
        &mut self,
        node: H::Node,
        circ: &Circuit<H>,
        output_param: impl FnMut(usize, &[String]) -> Option<String>,
        make_operation: impl FnOnce(usize, usize, &[String]) -> tket_json_rs::circuit_json::Operation,
    ) -> Result<(), Tk1ConvertError<H::Node>> {
        let TrackedValues {
            mut qubits,
            mut bits,
            params,
        } = self.get_input_values(node, circ)?;
        let params: Vec<String> = params
            .into_iter()
            .map(|p| self.values.param_expression(p).to_owned())
            .collect();

        // Update the values in the node's outputs.
        //
        // We preserve the order of linear values in the input.
        let mut qubit_iterator = qubits.iter().copied();
        let new_outputs = self.register_node_outputs(
            node,
            circ,
            &mut qubit_iterator,
            &params,
            output_param,
            |_| true,
        )?;
        qubits.extend(new_outputs.qubits);
        bits.extend(new_outputs.bits);

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

    /// Helper to emit a node that transparently forwards its inputs to its
    /// outputs, resulting in no pytket operation.
    ///
    /// It registers the node's input qubits and bits to the output
    /// wires, without modifying the tket1 circuit being constructed.
    /// Output parameters are more flexible, and output expressions can be
    /// computed from the input parameters via the `output_param` function.
    ///
    /// The node's inputs should have exactly the same number of qubits and
    /// bits. This method will return an error if that is not the case.
    ///
    /// You must also ensure that all input and output types are supported by
    /// the encoder. Otherwise, the function will return an error.
    ///
    /// ## Arguments
    ///
    /// - `node`: The HUGR for which to emit the command. Qubits and bits are
    ///   automatically retrieved from the node's inputs/outputs.
    /// - `circ`: The circuit containing the node.
    /// - `output_param`: A function that given a parameter index and the list
    ///   of input parameter values returns the string expression to use for the
    ///   parameter output. Returning `None` will abort the encoding.
    pub fn emit_transparent_node(
        &mut self,
        node: H::Node,
        circ: &Circuit<H>,
        mut output_param: impl FnMut(usize, &[String]) -> Option<String>,
    ) -> Result<(), Tk1ConvertError<H::Node>> {
        // Now we can gather all inputs and assign them to the node outputs transparently.
        //
        // Once we call `get_input_values`, the wires get marked as explored so if we find
        // in the rest of the function that we cannot emit the node, we will not be able to
        // revert the state of the tracker and must return an error instead.
        let input_values = self.get_input_values(node, circ)?;
        let mut qubits = input_values.qubits.into_iter();
        let mut bits = input_values.bits.into_iter();

        let params: Vec<String> = input_values
            .params
            .into_iter()
            .map(|p| self.values.param_expression(p).to_owned())
            .collect_vec();

        let op = circ.hugr().get_optype(node);
        let signature = op.dataflow_signature();
        let static_output = op.static_output_port();
        let other_output = op.other_output_port();
        for out_port in circ.hugr().node_outputs(node) {
            let ty = if Some(out_port) == other_output {
                continue;
            } else if Some(out_port) == static_output {
                let EdgeKind::Const(ty) = op.static_output().unwrap() else {
                    return Err(Tk1ConvertError::custom(format!(
                        "Cannot emit a static output for a {}.",
                        op.name()
                    )));
                };
                ty
            } else {
                let Some(ty) = signature
                    .as_ref()
                    .and_then(|s| s.out_port_type(out_port).cloned())
                else {
                    return Err(Tk1ConvertError::custom(
                        "Cannot emit a transparent node without a dataflow signature.",
                    ));
                };
                ty
            };

            let wire = hugr::Wire::new(node, out_port);
            let Some(count) = self.config().type_to_pytket(&ty)? else {
                return Err(Tk1ConvertError::custom(format!(
                    "Found an unsupported type while encoding a {}.",
                    op.name()
                )));
            };
            let mut values: Vec<TrackedValue> = Vec::with_capacity(count.total());
            for _ in 0..count.qubits {
                let Some(qb) = qubits.next() else {
                    return Err(Tk1ConvertError::custom(format!(
                        "Not enough qubits in the input values for a {}.",
                        op.name()
                    )));
                };
                values.push(qb.into());
            }
            for _ in 0..count.bits {
                let Some(bit) = bits.next() else {
                    return Err(Tk1ConvertError::custom(format!(
                        "Not enough bits in the input values for a {}.",
                        op.name()
                    )));
                };
                values.push(bit.into());
            }
            for i in 0..count.params {
                let Some(out) = output_param(i, &params) else {
                    return Err(Tk1ConvertError::custom(format!(
                        "Cannot encode output parameter #{i} for a {}.",
                        op.name()
                    )));
                };
                let p = self.values.new_param(out);
                values.push(p.into());
            }
            self.values.register_wire(wire, values, circ)?;
        }

        Ok(())
    }

    /// Helper to emit a new tket1 command corresponding to subgraph of unsupported nodes,
    /// encoded inside a pytket barrier.
    ///
    /// ## Arguments
    ///
    /// - `unsupported_nodes`: The list of nodes to encode as an unsupported subgraph.
    fn emit_unsupported(
        &mut self,
        unsupported_nodes: BTreeSet<H::Node>,
        circ: &Circuit<H>,
    ) -> Result<(), Tk1ConvertError<H::Node>> {
        let subcircuit_id = format!("tk{}", unsupported_nodes.iter().min().unwrap());

        // TODO: Use a cached topo checker here instead of traversing the full graph each time we create a `SiblingSubgraph`.
        //
        // TopoConvexChecker likes to borrow the hugr, so it'd be too invasive to store in the `Context`.
        let subgraph = SiblingSubgraph::try_from_nodes(
            unsupported_nodes.iter().cloned().collect_vec(),
            circ.hugr(),
        )
        .unwrap_or_else(|_| {
            panic!(
                "Failed to create subgraph from unsupported nodes [{}]",
                unsupported_nodes.iter().join(", ")
            )
        });
        let input_nodes: HashSet<_> = subgraph
            .incoming_ports()
            .iter()
            .flat_map(|inp| inp.iter().map(|(n, _)| *n))
            .collect();
        let output_nodes: HashSet<_> = subgraph.outgoing_ports().iter().map(|(n, _)| *n).collect();

        let unsupported_hugr = subgraph.extract_subgraph(circ.hugr(), &subcircuit_id);
        let payload = Package::from_hugr(unsupported_hugr)
            .unwrap()
            .store_str(EnvelopeConfig::text())
            .unwrap();

        // Collects the input values for the subgraph.
        //
        // The [`UnsupportedTracker`] ensures that at this point all input wires must come from
        // already-encoded nodes, and not from other unsupported nodes not in `unsupported_nodes`.
        let mut op_values = TrackedValues::default();
        for node in &input_nodes {
            let node_vals = self.get_input_values_internal(*node, circ, |w| {
                unsupported_nodes.contains(&w.node())
            })?;
            op_values.append(node_vals);
        }
        let input_param_exprs: Vec<String> = std::mem::take(&mut op_values.params)
            .into_iter()
            .map(|p| self.values.param_expression(p).to_owned())
            .collect();

        // Update the values in the node's outputs, and extend `op_values` with
        // any new output values.
        //
        // Output parameters are mapped to a fresh variable, that can be tracked
        // back to the encoded subcircuit's function name.
        let mut input_qubits = op_values.qubits.clone().into_iter();
        for &node in &output_nodes {
            let new_outputs = self.register_node_outputs(
                node,
                circ,
                &mut input_qubits,
                &[],
                |i, _| Some(format!("{subcircuit_id}_out{i}")),
                |_| true,
            )?;
            op_values.append(new_outputs);
        }

        // Create pytket operation, and add the subcircuit as hugr
        let mut tk1_op = make_tk1_operation(
            tket_json_rs::OpType::Barrier,
            op_values.qubits.len(),
            op_values.bits.len(),
            &input_param_exprs,
        );
        tk1_op.data = Some(payload);

        let opgroup = Some("tket2".to_string());
        self.emit_command(tk1_op, &op_values.qubits, &op_values.bits, opgroup);
        Ok(())
    }

    /// Emit a new tket1 command.
    ///
    /// This is a general-purpose command that can be used to emit any tket1
    /// operation, not necessarily corresponding to a specific HUGR node.
    ///
    /// Ensure that any output wires from the node being processed gets the
    /// appropriate values registered by calling [`ValueTracker::register_wire`]
    /// on the context's [`Tk1EncoderContext::values`] tracker.
    ///
    /// In general you should prefer using [`Tk1EncoderContext::emit_node`] or
    /// [`Tk1EncoderContext::emit_node_with_out_params`] as they automatically
    /// compute the input qubits and bits from the HUGR node, and ensure that
    /// output wires get their new values registered on the tracker.
    ///
    /// ## Arguments
    ///
    /// - `tk1_operation`: The tket1 operation to emit. See
    ///   [`make_tk1_operation`] for a helper function to create it.
    /// - `qubits`: The qubit registers to use as inputs/outputs of the pytket
    ///   op. Normally obtained from a HUGR node's inputs using
    ///   [`Tk1EncoderContext::get_input_values`] or allocated via
    ///   [`ValueTracker::new_qubit`].
    /// - `bits`: The bit registers to use as inputs/outputs of the pytket op.
    ///   Normally obtained from a HUGR node's inputs using
    ///   [`Tk1EncoderContext::get_input_values`] or allocated via
    ///   [`ValueTracker::new_bit`].
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
            OpType::LoadConstant(_) => {
                self.emit_transparent_node(node, circ, |i, ps| Some(ps[i].clone()))?;
                return Ok(true);
            }
            OpType::Const(op) => {
                let config = Arc::clone(&self.config);
                if let Some(values) = config.const_to_pytket(&op.value, self)? {
                    let wire = Wire::new(node, 0);
                    self.values.register_wire(wire, values.into_iter(), circ)?;
                    return Ok(true);
                }
            }
            _ => {}
        }

        self.unsupported.record_node(node, circ);
        Ok(false)
    }

    /// Helper to register values for a node's output wires.
    ///
    /// Returns any new value associated with the output wires.
    ///
    /// ## Arguments
    ///
    /// - `node`: The node to register the outputs for.
    /// - `circ`: The circuit containing the node.
    /// - `qubit_values`: An iterator of existing qubit ids to re-use for the output.
    ///   Once all qubits have been used, new qubit ids will be generated.
    /// - `input_params`: The list of input parameter expressions.
    /// - `output_param`: A function that takes the index of the output parameter
    ///   and the list of input parameter expressions, and returns the string
    ///   expression to use for the output parameter.
    /// - `wire_filter`: A function that takes a wire and returns true if the wire
    ///   at the output of the `node` should be registered.
    fn register_node_outputs(
        &mut self,
        node: H::Node,
        circ: &Circuit<H>,
        qubit_values: &mut impl Iterator<Item = TrackedQubit>,
        input_params: &[String],
        mut output_param: impl FnMut(usize, &[String]) -> Option<String>,
        wire_filter: impl Fn(Wire<H::Node>) -> bool,
    ) -> Result<TrackedValues, Tk1ConvertError<H::Node>> {
        // Update the values in the node's outputs.
        //
        // We preserve the order of linear values in the input
        let optype = circ.hugr().get_optype(node);
        let sig = optype.dataflow_signature();
        let other_output_port = optype.other_output_port();
        let mut new_outputs = TrackedValues::default();
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
            if !wire_filter(out_wire) {
                continue;
            }

            let mut out_wire_values = Vec::with_capacity(type_counts.total());
            for _ in 0..type_counts.qubits {
                let qb = qubit_values.next().unwrap_or_else(|| {
                    // If we already have a matching output for all input qubits,
                    // generate a fresh ID.
                    let qb = self.values.new_qubit();
                    new_outputs.qubits.push(qb);
                    qb
                });
                out_wire_values.push(TrackedValue::Qubit(qb));
            }
            for _ in 0..type_counts.bits {
                let b = self.values.new_bit();
                new_outputs.bits.push(b);
                out_wire_values.push(TrackedValue::Bit(b));
            }
            for _ in 0..type_counts.params {
                let out_param_expression =
                    output_param(new_outputs.params.len(), input_params).unwrap_or("0".into());
                let p = self.values.new_param(out_param_expression);
                new_outputs.params.push(p);
                out_wire_values.push(TrackedValue::Param(p));
            }
            self.values.register_wire(out_wire, out_wire_values, circ)?;
        }

        Ok(new_outputs)
    }
}

/// Initialize a tket1 [Operation](circuit_json::Operation) to pass to
/// [`Tk1EncoderContext::emit_command`].
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
