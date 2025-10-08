//! Intermediate structure for encoding [`Circuit`]s into [`SerialCircuit`]s.

mod unit_generator;
mod unsupported_tracker;
mod value_tracker;

use hugr::core::HugrNode;
use hugr::envelope::EnvelopeConfig;
use hugr::hugr::views::SiblingSubgraph;
use hugr::package::Package;
use hugr_core::hugr::internal::PortgraphNodeMap;
use tket_json_rs::clexpr::InputClRegister;
use tket_json_rs::opbox::BoxID;
pub use value_tracker::{
    TrackedBit, TrackedParam, TrackedQubit, TrackedValue, TrackedValues, ValueTracker,
};

use hugr::ops::{OpTrait, OpType};
use hugr::types::EdgeKind;

use std::borrow::Cow;
use std::collections::{BTreeSet, HashMap, HashSet};
use std::sync::{Arc, RwLock};

use hugr::{HugrView, Wire};
use itertools::Itertools;
use tket_json_rs::circuit_json::{self, SerialCircuit};
use unsupported_tracker::UnsupportedTracker;

use super::{
    OpConvertError, PytketEncodeError, METADATA_OPGROUP, METADATA_PHASE, METADATA_Q_REGISTERS,
};
use crate::circuit::Circuit;
use crate::serialize::pytket::config::PytketEncoderConfig;
use crate::serialize::pytket::extension::RegisterCount;

/// The state of an in-progress [`SerialCircuit`] being built from a [`Circuit`].
#[derive(derive_more::Debug)]
#[debug(bounds(H: HugrView))]
pub struct PytketEncoderContext<H: HugrView> {
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
    config: Arc<PytketEncoderConfig<H>>,
    /// A cache of translated hugr functions, to be encoded as op boxes.
    function_cache: Arc<RwLock<HashMap<H::Node, CachedEncodedFunction>>>,
}

/// Options used when emitting a pytket command from HUGR operations.
///
/// Mostly related to qubit/bit/parameter reuse.
#[derive(Default)]
#[allow(clippy::type_complexity)]
pub struct EmitCommandOptions<'a> {
    /// A function returning a list of input qubits to reuse in the output.
    /// Any additional required qubits IDs will be freshly generated.
    ///
    /// If not provided, input qubits will be reused in the order they appear in the input.
    reuse_qubits_fn: Option<Box<dyn FnOnce(&[TrackedQubit]) -> Vec<TrackedQubit> + 'a>>,
    /// A function returning a list of input bits to reuse in the output.
    /// Any additional required bits IDs will be freshly generated.
    ///
    /// If not provided, only fresh bit IDs will be used.
    reuse_bits_fn: Option<Box<dyn FnOnce(&[TrackedBit]) -> Vec<TrackedBit> + 'a>>,
    /// A function that computes the command's output parameters, given the
    /// input expressions.
    ///
    /// If the number of parameters does not match the expected number, the
    /// encoding result in an error.
    ///
    /// If not provided, no output parameters will be computed.
    output_params_fn: Option<Box<dyn FnOnce(OutputParamArgs<'_>) -> Vec<String> + 'a>>,
}

impl<'a> EmitCommandOptions<'a> {
    /// Create a new [`EmitCommandOptions`] with the default values.
    pub fn new() -> Self {
        Self {
            reuse_qubits_fn: None,
            reuse_bits_fn: None,
            output_params_fn: None,
        }
    }

    /// Set a function returning a list of input qubits to reuse in the output.
    ///
    /// Overrides the default behaviour of reusing input qubits in the order they appear in the input.
    pub fn reuse_qubits(
        mut self,
        reuse_qubits: impl FnOnce(&[TrackedQubit]) -> Vec<TrackedQubit> + 'a,
    ) -> Self {
        self.reuse_qubits_fn = Some(Box::new(reuse_qubits));
        self
    }

    /// Set a function returning a list of input bits to reuse in the output.
    ///
    /// Overrides the default behaviour of only using fresh bit IDs.
    pub fn reuse_bits(
        mut self,
        reuse_bits: impl FnOnce(&[TrackedBit]) -> Vec<TrackedBit> + 'a,
    ) -> Self {
        self.reuse_bits_fn = Some(Box::new(reuse_bits));
        self
    }

    /// Reuse all input bits in the output, in the order they appear in the input.
    pub fn reuse_all_bits(self) -> Self {
        self.reuse_bits(|inp_bits| inp_bits.to_owned())
    }

    /// Set a function that computes the command's output parameters, given the
    /// input expressions.
    ///
    /// Overrides the default behaviour of not computing output parameters.
    pub fn output_params(
        mut self,
        output_params: impl FnOnce(OutputParamArgs<'_>) -> Vec<String> + 'a,
    ) -> Self {
        self.output_params_fn = Some(Box::new(output_params));
        self
    }
}

impl<H: HugrView> PytketEncoderContext<H> {
    /// Create a new [`PytketEncoderContext`] from a [`Circuit`].
    pub(super) fn new(
        circ: &Circuit<H>,
        region: H::Node,
        config: impl Into<Arc<PytketEncoderConfig<H>>>,
    ) -> Result<Self, PytketEncodeError<H::Node>> {
        let config: Arc<PytketEncoderConfig<H>> = config.into();
        let hugr = circ.hugr();
        let name = Circuit::new(hugr.with_entrypoint(region))
            .name()
            .map(str::to_string)
            .filter(|s| !s.is_empty());

        // Recover other parameters stored in the metadata
        let phase = match hugr.get_metadata(region, METADATA_PHASE) {
            Some(p) => p.as_str().unwrap().to_string(),
            None => "0".to_string(),
        };

        Ok(Self {
            name,
            phase,
            commands: vec![],
            values: ValueTracker::new(circ, region, &config)?,
            unsupported: UnsupportedTracker::new(circ),
            config,
            function_cache: Arc::new(RwLock::new(HashMap::new())),
        })
    }

    /// Traverse the circuit region in topological order, encoding the nodes as
    /// pytket commands.
    ///
    /// Returns the final [`SerialCircuit`] if successful.
    pub(super) fn run_encoder(
        &mut self,
        circ: &Circuit<H>,
        region: H::Node,
    ) -> Result<(), PytketEncodeError<H::Node>> {
        // When encoding a function, mark it as being encoded to detect recursive calls.
        if circ.hugr().get_parent(region) == Some(circ.hugr().module_root()) {
            let Ok(mut cache) = self.function_cache.write() else {
                // If the cache is poisoned, some thread has panicked while holding the lock.
                return Err(PytketEncodeError::custom("Detected encoder worker panic."));
            };
            cache.insert(region, CachedEncodedFunction::InEncodingStack);
        }

        let (region, node_map) = circ.hugr().region_portgraph(region);
        // TODO: Use weighted topological sort to try and explore unsupported
        // ops first (that is, ops with no available emitter in `self.config`),
        // to ensure we group them as much as possible.
        let mut topo = petgraph::visit::Topo::new(&region);
        while let Some(pg_node) = topo.next(&region) {
            let node = node_map.from_portgraph(pg_node);
            self.try_encode_node(node, circ)?;
        }
        Ok(())
    }

    /// Finish building and return the final [`SerialCircuit`],
    /// as well as any parameter expressions at the circuit's output.
    pub(super) fn finish(
        mut self,
        circ: &Circuit<H>,
        region: H::Node,
    ) -> Result<(SerialCircuit, Vec<String>), PytketEncodeError<H::Node>> {
        // Add any remaining unsupported nodes
        //
        // TODO: Test that unsupported subgraphs that don't affect any qubit/bit registers
        // are correctly encoded in pytket commands.
        while !self.unsupported.is_empty() {
            let node = self.unsupported.iter().next().unwrap();
            let unsupported_subgraph = self.unsupported.extract_component(node);
            self.emit_unsupported(unsupported_subgraph, circ)?;
        }

        let final_values = self.values.finish(circ, region)?;

        let mut ser = SerialCircuit::new(self.name, self.phase);

        ser.commands = self.commands;
        ser.qubits = final_values.qubits.into_iter().map_into().collect();
        ser.bits = final_values.bits.into_iter().map_into().collect();
        ser.implicit_permutation = final_values.qubit_permutation;
        ser.number_of_ws = None;
        Ok((ser, final_values.params))
    }

    /// Returns a reference to this encoder's configuration.
    pub fn config(&self) -> &PytketEncoderConfig<H> {
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
    /// supported, use [`PytketEncoderConfig::type_to_pytket`] on the encoder
    /// context's [`PytketEncoderContext::config`].
    ///
    /// ### Errors
    ///
    /// - [`OpConvertError::WireHasNoValues`] if the wire is not tracked or has
    ///   a type that cannot be converted to pytket values.
    pub fn get_wire_values(
        &mut self,
        wire: Wire<H::Node>,
        circ: &Circuit<H>,
    ) -> Result<Cow<'_, [TrackedValue]>, PytketEncodeError<H::Node>> {
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
    /// These values can be used with the [`PytketEncoderContext::values`]
    /// tracker to retrieve the corresponding pytket registers and parameter
    /// expressions. See [`ValueTracker::qubit_register`],
    /// [`ValueTracker::bit_register`], and [`ValueTracker::param_expression`].
    ///
    /// Marks the input connections to the node as explored. When all ports
    /// connected to a wire have been explored, the wire is removed from the
    /// tracker.
    ///
    /// If an input wire is the output of an unsupported node, a subgraph of
    /// unsupported nodes containing it will be emitted as a pytket barrier.
    ///
    /// This function SHOULD NOT be called before determining that the node
    /// operation is supported, as it will mark the input connections as explored
    /// and potentially remove them from the tracker. To determine if a node
    /// operation is supported, use [`PytketEncoderConfig::type_to_pytket`] on
    /// the encoder context's [`PytketEncoderContext::config`].
    pub fn get_input_values(
        &mut self,
        node: H::Node,
        circ: &Circuit<H>,
    ) -> Result<TrackedValues, PytketEncodeError<H::Node>> {
        self.get_input_values_internal(node, circ, |_| true)
            .try_into_tracked_values()
    }

    /// Auxiliary function used to collect the input values of a node.
    /// See [`PytketEncoderContext::get_input_values`].
    ///
    /// Given a node in the HUGR, returns all the [`TrackedValue`]s associated
    /// with its inputs. Calls `wire_filter` to decide which incoming wires to include.
    fn get_input_values_internal(
        &mut self,
        node: H::Node,
        circ: &Circuit<H>,
        wire_filter: impl Fn(Wire<H::Node>) -> bool,
    ) -> NodeInputValues<H::Node> {
        let mut tracked_values = TrackedValues::default();
        let mut unknown_values = Vec::new();

        let optype = circ.hugr().get_optype(node);
        let other_input_port = optype.other_input_port();
        for input in circ.hugr().node_inputs(node) {
            // Ignore order edges.
            if Some(input) == other_input_port {
                continue;
            }
            // Dataflow ports should have a single linked neighbour.
            let (neigh, neigh_out) = circ
                .hugr()
                .single_linked_output(node, input)
                .expect("Dataflow input port should have a single neighbour");
            let wire = Wire::new(neigh, neigh_out);
            if !wire_filter(wire) {
                continue;
            }

            match self.get_wire_values(wire, circ) {
                Ok(values) => tracked_values.extend(values.iter().copied()),
                Err(PytketEncodeError::OpConversionError(OpConvertError::WireHasNoValues {
                    wire,
                })) => unknown_values.push(wire),
                Err(e) => panic!(
                    "get_wire_values should only return WireHasNoValues errors, but got: {e}"
                ),
            }
        }
        NodeInputValues {
            tracked_values,
            unknown_values,
        }
    }

    /// Helper to emit a new tket1 command corresponding to a single HUGR node.
    ///
    /// See [`EmitCommandOptions`] for controlling the output qubit, bits, and
    /// parameter expressions.
    ///
    /// See [`PytketEncoderContext::emit_command`] for more general cases where
    /// commands are not associated to a specific node.
    ///
    /// ## Arguments
    ///
    /// - `pytket_optype`: The tket1 operation type to emit.
    /// - `node`: The HUGR node for which to emit the command. Qubits and bits
    ///   are automatically retrieved from the node's inputs/outputs.
    /// - `circ`: The circuit containing the node.
    /// - `options`: Options for controlling the output qubit, bits, and
    ///   parameter expressions.
    pub fn emit_node(
        &mut self,
        pytket_optype: tket_json_rs::OpType,
        node: H::Node,
        circ: &Circuit<H>,
        options: EmitCommandOptions,
    ) -> Result<(), PytketEncodeError<H::Node>> {
        self.emit_node_command(node, circ, options, move |inputs| {
            make_tk1_operation(pytket_optype, inputs)
        })
    }

    /// Helper to emit a new tket1 command corresponding to a single HUGR node,
    /// using a custom operation generator and computing output parameter
    /// expressions. Use [`PytketEncoderContext::emit_node`] when pytket operation
    /// can be defined directly from a [`tket_json_rs::OpType`].
    ///
    /// See [`PytketEncoderContext::emit_command`] for a general case emitter.
    ///
    /// ## Arguments
    ///
    /// - `node`: The HUGR node for which to emit the command. Qubits and bits
    ///   are automatically retrieved from the node's inputs/outputs. Input
    ///   arguments are listed in order, followed by output-only args.
    /// - `circ`: The circuit containing the node.
    /// - `reuse_bits`: A function returning a lits of input bits to reuse in the output.
    ///   Any additional required bits IDs will be freshly generated.
    /// - `output_params`: A function that computes the output parameter
    ///   expressions from the list of input parameters. If the number of
    ///   parameters does not match the expected number, the encoding will fail.
    /// - `make_operation`: A function that takes the number of qubits, bits,
    ///   and the list of input parameter expressions and returns a pytket
    ///   operation. See [`make_tk1_operation`] for a helper function to create
    ///   it.
    pub fn emit_node_command(
        &mut self,
        node: H::Node,
        circ: &Circuit<H>,
        options: EmitCommandOptions,
        make_operation: impl FnOnce(MakeOperationArgs<'_>) -> tket_json_rs::circuit_json::Operation,
    ) -> Result<(), PytketEncodeError<H::Node>> {
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
        let new_outputs =
            self.register_node_outputs(node, circ, &qubits, &bits, &params, options, |_| true)?;
        qubits.extend(new_outputs.qubits);
        bits.extend(new_outputs.bits);

        // Preserve the pytket opgroup, if it got stored in the metadata.
        let opgroup: Option<String> = circ
            .hugr()
            .get_metadata(node, METADATA_OPGROUP)
            .and_then(serde_json::Value::as_str)
            .map(ToString::to_string);

        let args = MakeOperationArgs {
            num_qubits: qubits.len(),
            num_bits: bits.len(),
            params: Cow::Borrowed(&params),
        };
        let op = make_operation(args);

        self.emit_command(op, &qubits, &bits, opgroup);
        Ok(())
    }

    /// Helper to emit a node that transparently forwards its inputs to its
    /// outputs, resulting in no pytket operation.
    ///
    /// It registers the node's input qubits and bits to the output
    /// wires, without modifying the tket1 circuit being constructed.
    /// Output parameters are more flexible, and output expressions can be
    /// computed from the input parameters via the `output_params` function.
    ///
    /// The node's inputs should have exactly the same number of qubits and
    /// bits. This method will return an error if that is not the case.
    ///
    /// You must also ensure that all input and output types are supported by
    /// the encoder. Otherwise, the function will return an error.
    ///
    /// ## Arguments
    ///
    /// - `node`: The HUGR node for which to emit the command. Qubits and bits are
    ///   automatically retrieved from the node's inputs/outputs.
    /// - `circ`: The circuit containing the node.
    /// - `output_params`: A function that computes the output parameter
    ///   expressions from the list of input parameters. If the number of parameters
    ///   does not match the expected number, the encoding will fail.
    pub fn emit_transparent_node(
        &mut self,
        node: H::Node,
        circ: &Circuit<H>,
        output_params: impl FnOnce(OutputParamArgs<'_>) -> Vec<String>,
    ) -> Result<(), PytketEncodeError<H::Node>> {
        let input_values = self.get_input_values(node, circ)?;
        let output_counts = self.node_output_values(node, circ)?;
        let total_out_count: RegisterCount = output_counts.iter().map(|(_, c)| *c).sum();

        // Compute all the output parameters at once
        let input_params: Vec<String> = input_values
            .params
            .into_iter()
            .map(|p| self.values.param_expression(p).to_owned())
            .collect_vec();
        let out_params = output_params(OutputParamArgs {
            expected_count: total_out_count.params,
            input_params: &input_params,
        });

        // Check that we got the expected number of outputs.
        if input_values.qubits.len() != total_out_count.qubits {
            return Err(PytketEncodeError::custom(format!(
                "Mismatched number of input and output qubits while trying to emit a transparent operation for {}. We have {} inputs but {} outputs.",
                circ.hugr().get_optype(node),
                input_values.qubits.len(),
                total_out_count.qubits,
            )));
        }
        if input_values.bits.len() != total_out_count.bits {
            return Err(PytketEncodeError::custom(format!(
                "Mismatched number of input and output bits while trying to emit a transparent operation for {}. We have {} inputs but {} outputs.",
                circ.hugr().get_optype(node),
                input_values.bits.len(),
                total_out_count.bits,
            )));
        }
        if out_params.len() != total_out_count.params {
            return Err(PytketEncodeError::custom(format!(
                "Expected {} parameters in the input values for a {}, but got {}.",
                total_out_count.params,
                circ.hugr().get_optype(node),
                out_params.len()
            )));
        }

        // Now we can gather all inputs and assign them to the node outputs transparently.
        let mut qubits = input_values.qubits.into_iter();
        let mut bits = input_values.bits.into_iter();
        let mut params = out_params.into_iter();
        for (wire, count) in output_counts {
            let mut values: Vec<TrackedValue> = Vec::with_capacity(count.total());
            values.extend(qubits.by_ref().take(count.qubits).map(TrackedValue::Qubit));
            values.extend(bits.by_ref().take(count.bits).map(TrackedValue::Bit));
            for p in params.by_ref().take(count.params) {
                values.push(self.values.new_param(p).into());
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
    ) -> Result<(), PytketEncodeError<H::Node>> {
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
            .store_str(EnvelopeConfig::text())
            .unwrap();

        // Collects the input values for the subgraph.
        //
        // The [`UnsupportedTracker`] ensures that at this point all local input wires must come from
        // already-encoded nodes, and not from other unsupported nodes not in `unsupported_nodes`.
        //
        // Non-local incoming edges (e.g. function references) must be marked so they can be recovered when
        // decoding the circuit.
        //
        // TODO: We must encode additional metadata for external function calls.
        // For now we just ignore them.
        let mut op_values = TrackedValues::default();
        let mut external_edges = Vec::new();
        for node in &input_nodes {
            let NodeInputValues {
                tracked_values,
                unknown_values,
            } = self
                .get_input_values_internal(*node, circ, |w| !unsupported_nodes.contains(&w.node()));
            op_values.append(tracked_values);
            external_edges.extend(unknown_values);
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
        for &node in &output_nodes {
            let new_outputs = self.register_node_outputs(
                node,
                circ,
                &op_values.qubits,
                &op_values.bits,
                &input_param_exprs,
                EmitCommandOptions::new().output_params(|p| {
                    (0..p.expected_count)
                        .map(|i| format!("{subcircuit_id}_out{i}"))
                        .collect_vec()
                }),
                |_| true,
            )?;
            op_values.append(new_outputs);
        }

        // Create pytket operation, and add the subcircuit as hugr
        let args = MakeOperationArgs {
            num_qubits: op_values.qubits.len(),
            num_bits: op_values.bits.len(),
            params: Cow::Borrowed(&input_param_exprs),
        };
        let mut pytket_op = make_tk1_operation(tket_json_rs::OpType::Barrier, args);
        pytket_op.data = Some(payload);

        let opgroup = Some("UNSUPPORTED_HUGR".to_string());
        self.emit_command(pytket_op, &op_values.qubits, &op_values.bits, opgroup);
        Ok(())
    }

    /// Emit a new tket1 command.
    ///
    /// This is a general-purpose command that can be used to emit any tket1
    /// operation, not necessarily corresponding to a specific HUGR node.
    ///
    /// Ensure that any output wires from the node being processed gets the
    /// appropriate values registered by calling [`ValueTracker::register_wire`]
    /// on the context's [`PytketEncoderContext::values`] tracker.
    ///
    /// In general you should prefer using [`PytketEncoderContext::emit_node`]
    /// as it automatically computes the input qubits and bits from the HUGR
    /// node, and ensure that output wires get their new values registered on
    /// the tracker.
    ///
    /// ## Arguments
    ///
    /// - `pytket_op`: The tket1 operation to emit. See [`make_tk1_operation`]
    ///   for a helper function to create it.
    /// - `qubits`: The qubit registers to use as inputs/outputs of the pytket
    ///   op. Normally obtained from a HUGR node's inputs using
    ///   [`PytketEncoderContext::get_input_values`] or allocated via
    ///   [`ValueTracker::new_qubit`].
    /// - `bits`: The bit registers to use as inputs/outputs of the pytket op.
    ///   Normally obtained from a HUGR node's inputs using
    ///   [`PytketEncoderContext::get_input_values`] or allocated via
    ///   [`ValueTracker::new_bit`].
    /// - `opgroup`: A tket1 operation group identifier, if any.
    pub fn emit_command(
        &mut self,
        pytket_op: circuit_json::Operation,
        qubits: &[TrackedQubit],
        bits: &[TrackedBit],
        opgroup: Option<String>,
    ) {
        let qubit_regs = qubits.iter().map(|&qb| self.values.qubit_register(qb));
        let bit_regs = bits.iter().map(|&b| self.values.bit_register(b));
        let command = circuit_json::Command {
            op: pytket_op,
            args: qubit_regs.chain(bit_regs).cloned().collect(),
            opgroup,
        };

        self.commands.push(command);
    }

    /// Helper to emit a `CircBox` tket1 command corresponding to a region of the Hugr.
    ///
    /// Returns a bool indicating whether the subcircuit was successfully emitted,
    /// or should be encoded opaquely instead. This is the case when the subcircuit
    /// contains output parameters.
    ///
    // TODO: Support output parameters in subcircuits. This may require
    // substituting variables in the parameter expressions.
    fn emit_subcircuit(
        &mut self,
        node: H::Node,
        circ: &Circuit<H>,
    ) -> Result<EncodeStatus, PytketEncodeError<H::Node>> {
        let config = Arc::clone(&self.config);

        // Recursively encode the sub-graph.
        let mut subencoder = PytketEncoderContext::new(circ, node, config)?;
        subencoder.function_cache = self.function_cache.clone();
        subencoder.run_encoder(circ, node)?;

        let (serial_subcirc, output_params) = subencoder.finish(circ, node)?;
        if !output_params.is_empty() {
            return Ok(EncodeStatus::Unsupported);
        }

        self.emit_circ_box(node, serial_subcirc, circ)?;
        Ok(EncodeStatus::Success)
    }

    /// Helper to emit a `CircBox` tket1 command corresponding to a function definition in the Hugr.
    ///
    /// The function encoding is cached and reused if possible.
    ///
    /// Returns a bool indicating whether the subcircuit was successfully emitted,
    /// or should be encoded opaquely instead. This is the case when the subcircuit
    /// contains output parameters.
    ///
    // TODO: Support output parameters in subcircuits. This may require
    // substituting variables in the parameter expressions.
    fn emit_function_call(
        &mut self,
        node: H::Node,
        function: H::Node,
        circ: &Circuit<H>,
    ) -> Result<EncodeStatus, PytketEncodeError<H::Node>> {
        let cache = self.function_cache.read().ok();
        if let Some(encoded) = cache.as_ref().and_then(|c| c.get(&function)) {
            let encoded = encoded.clone();
            drop(cache);
            match encoded {
                CachedEncodedFunction::Encoded { serial_circuit } => {
                    self.emit_circ_box(node, serial_circuit, circ)?;
                    return Ok(EncodeStatus::Success);
                }
                CachedEncodedFunction::Unsupported | CachedEncodedFunction::InEncodingStack => {
                    return Ok(EncodeStatus::Unsupported);
                }
            };
        }
        drop(cache);

        // If the function is not cached, we need to encode it.
        let config = Arc::clone(&self.config);

        // Recursively encode the sub-graph.
        let mut subencoder = PytketEncoderContext::new(circ, function, config)?;
        subencoder.function_cache = self.function_cache.clone();
        subencoder.run_encoder(circ, function)?;
        let (serial_subcirc, output_params) = subencoder.finish(circ, function)?;

        let (result, cached_fn) = match output_params.is_empty() {
            true => (
                EncodeStatus::Success,
                CachedEncodedFunction::Encoded {
                    serial_circuit: serial_subcirc.clone(),
                },
            ),
            false => (
                EncodeStatus::Unsupported,
                CachedEncodedFunction::Unsupported,
            ),
        };

        // Cache the encoded subcircuit for future use.
        // If the cache is poisoned, ignore it.
        if let Ok(mut cache) = self.function_cache.write() {
            cache.insert(function, cached_fn);
        }

        if result == EncodeStatus::Success {
            self.emit_circ_box(node, serial_subcirc, circ)?;
        }
        Ok(result)
    }

    /// Helper to emit a `CircBox` tket1 command from a Serialised circuit.
    fn emit_circ_box(
        &mut self,
        node: H::Node,
        boxed_circuit: SerialCircuit,
        circ: &Circuit<H>,
    ) -> Result<(), PytketEncodeError<H::Node>> {
        self.emit_node_command(
            node,
            circ,
            EmitCommandOptions::new().reuse_all_bits(),
            |args| {
                let mut pytket_op = make_tk1_operation(tket_json_rs::OpType::CircBox, args);
                pytket_op.op_box = Some(tket_json_rs::opbox::OpBox::CircBox {
                    id: BoxID::new(),
                    circuit: boxed_circuit,
                });
                pytket_op
            },
        )?;
        Ok(())
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
    ) -> Result<EncodeStatus, PytketEncodeError<H::Node>> {
        let optype = circ.hugr().get_optype(node);

        if self.encode_nonlocal_inputs(node, optype, circ)? == EncodeStatus::Unsupported {
            self.unsupported.record_node(node, circ);
            return Ok(EncodeStatus::Unsupported);
        }

        // Try to encode the operation using each of the registered encoders.
        //
        // If none of the encoders can handle the operation, we just add it to
        // the unsupported tracker and move on.
        match optype {
            OpType::ExtensionOp(op) => {
                let config = Arc::clone(&self.config);
                if config.op_to_pytket(node, op, circ, self)? == EncodeStatus::Success {
                    return Ok(EncodeStatus::Success);
                }
            }
            OpType::LoadConstant(_) => {
                self.emit_transparent_node(node, circ, |ps| ps.input_params.to_owned())?;
                return Ok(EncodeStatus::Success);
            }
            OpType::Const(op) => {
                let config = Arc::clone(&self.config);
                if let Some(values) = config.const_to_pytket(&op.value, self)? {
                    let wire = Wire::new(node, 0);
                    self.values.register_wire(wire, values.into_iter(), circ)?;
                    return Ok(EncodeStatus::Success);
                }
            }
            OpType::DFG(_) => return self.emit_subcircuit(node, circ),
            OpType::Call(call) => {
                let (fn_node, _) = circ
                    .hugr()
                    .single_linked_output(node, call.called_function_port())
                    .expect("Function call must be linked to a function");
                if circ.hugr().get_optype(fn_node).is_func_defn()
                    && self.emit_function_call(node, fn_node, circ)? == EncodeStatus::Success
                {
                    return Ok(EncodeStatus::Success);
                }
            }
            OpType::Input(_) | OpType::Output(_) => {
                // I/O nodes are handled by the container's encoder.
                return Ok(EncodeStatus::Success);
            }
            _ => {}
        }

        self.unsupported.record_node(node, circ);
        Ok(EncodeStatus::Unsupported)
    }

    /// The toposort traversal in `run_encoder` only explores nodes in the
    /// region.
    ///
    /// When a node has a non-local input, we must process its originating node
    /// before trying to translate it.
    ///
    /// In general if a node has a non-local dataflow input we report it as unsupported,
    /// unless the input comes from a global definition that we are able to encode.
    ///
    /// # Returns
    ///
    /// - [`EncodeStatus::Success`] if all node inputs are supported.
    /// - [`EncodeStatus::Unsupported`] if the node has unsupported non-local dataflow inputs, and we should mark it as unsupported.
    fn encode_nonlocal_inputs(
        &mut self,
        node: H::Node,
        optype: &OpType,
        circ: &Circuit<H>,
    ) -> Result<EncodeStatus, PytketEncodeError<H::Node>> {
        let node_parent = circ.hugr().get_parent(node);

        // Explore the dataflow value and static inputs, but not the _other inputs_.
        let input_ports = circ
            .hugr()
            .node_inputs(node)
            .take(optype.value_input_count() + optype.static_input_port().is_some() as usize);

        for (neigh, neigh_port) in input_ports.flat_map(|inp| circ.hugr().linked_outputs(node, inp))
        {
            let wire = Wire::new(neigh, neigh_port);
            if self.values.peek_wire_values(wire).is_some() {
                // Ignore inputs that already have registered values.
                continue;
            }

            let neigh_parent = circ.hugr().get_parent(neigh);
            if neigh_parent == node_parent {
                continue;
            }
            if neigh_parent != Some(circ.hugr().module_root()) {
                // Non-global dataflow input, report as unsupported.
                return Ok(EncodeStatus::Unsupported);
            }
            let optype = circ.hugr().get_optype(neigh);
            match optype {
                OpType::FuncDefn(_) | OpType::FuncDecl(_) => {
                    // Function definitions/declarations have special handling to be able to encode Call nodes.
                    // We register them here with an empty set of values (since function-typed wires do not carry pytket values).
                    self.values
                        .register_wire::<TrackedValue>(wire, vec![], circ)?;
                }
                OpType::Const(_) => {
                    if self.try_encode_node(neigh, circ)? == EncodeStatus::Unsupported {
                        return Ok(EncodeStatus::Unsupported);
                    }
                }
                _ => {
                    return Ok(EncodeStatus::Unsupported);
                }
            }
        }
        Ok(EncodeStatus::Success)
    }

    /// Helper to register values for a node's output wires.
    ///
    /// Returns any new value associated with the output wires.
    ///
    /// ## Arguments
    ///
    /// - `node`: The node to register the outputs for.
    /// - `circ`: The circuit containing the node.
    /// - `input_qubits`: The qubit inputs to the operation.
    /// - `input_bits`: The bit inputs to the operation.
    /// - `input_params`: The list of input parameter expressions.
    /// - `options`: Options for controlling the output qubit, bits, and
    ///   parameter expressions.
    /// - `wire_filter`: A function that takes a wire and returns true if the wire
    ///   at the output of the `node` should be registered.
    #[allow(clippy::too_many_arguments)]
    fn register_node_outputs(
        &mut self,
        node: H::Node,
        circ: &Circuit<H>,
        input_qubits: &[TrackedQubit],
        input_bits: &[TrackedBit],
        input_params: &[String],
        options: EmitCommandOptions,
        wire_filter: impl Fn(Wire<H::Node>) -> bool,
    ) -> Result<TrackedValues, PytketEncodeError<H::Node>> {
        let output_counts = self.node_output_values(node, circ)?;
        let total_out_count: RegisterCount = output_counts.iter().map(|(_, c)| *c).sum();

        let output_qubits = match options.reuse_qubits_fn {
            Some(f) => f(input_qubits),
            None => input_qubits.to_vec(),
        };
        let output_bits = match options.reuse_bits_fn {
            Some(f) => f(input_bits),
            None => input_bits.to_vec(),
        };

        // Compute all the output parameters at once
        let out_params = match options.output_params_fn {
            Some(f) => f(OutputParamArgs {
                expected_count: total_out_count.params,
                input_params,
            }),
            None => Vec::new(),
        };

        // Check that we got the expected number of outputs.
        if out_params.len() != total_out_count.params {
            return Err(PytketEncodeError::custom(format!(
                "Expected {} parameters in the input values for a {}, but got {}.",
                total_out_count.params,
                circ.hugr().get_optype(node),
                out_params.len()
            )));
        }

        // Update the values in the node's outputs.
        //
        // We preserve the order of linear values in the input
        let mut qubits = output_qubits.iter().copied();
        let mut bits = output_bits.iter().copied();
        let mut params = out_params.into_iter();
        let mut new_outputs = TrackedValues::default();
        for (wire, count) in output_counts {
            if !wire_filter(wire) {
                continue;
            }

            let mut out_wire_values = Vec::with_capacity(count.total());

            // Qubits
            out_wire_values.extend(qubits.by_ref().take(count.qubits).map(TrackedValue::Qubit));
            for _ in out_wire_values.len()..count.qubits {
                // If we already assigned all input qubit ids, get a fresh one.
                let qb = self.values.new_qubit();
                new_outputs.qubits.push(qb);
                out_wire_values.push(TrackedValue::Qubit(qb));
            }

            // Bits
            let non_bit_count = out_wire_values.len();
            out_wire_values.extend(bits.by_ref().take(count.bits).map(TrackedValue::Bit));
            let reused_bit_count = out_wire_values.len() - non_bit_count;
            for _ in reused_bit_count..count.bits {
                let b = self.values.new_bit();
                new_outputs.bits.push(b);
                out_wire_values.push(TrackedValue::Bit(b));
            }

            // Parameters
            for expr in params.by_ref().take(count.params) {
                let p = self.values.new_param(expr);
                new_outputs.params.push(p);
                out_wire_values.push(p.into());
            }
            self.values.register_wire(wire, out_wire_values, circ)?;
        }

        Ok(new_outputs)
    }

    /// Return the output wires of a node that have an associated pytket [`RegisterCount`].
    #[allow(clippy::type_complexity)]
    fn node_output_values(
        &self,
        node: H::Node,
        circ: &Circuit<H>,
    ) -> Result<Vec<(Wire<H::Node>, RegisterCount)>, PytketEncodeError<H::Node>> {
        let op = circ.hugr().get_optype(node);
        let signature = op.dataflow_signature();
        let static_output = op.static_output_port();
        let other_output = op.other_output_port();
        let mut wire_counts = Vec::with_capacity(circ.hugr().num_outputs(node));
        for out_port in circ.hugr().node_outputs(node) {
            let ty = if Some(out_port) == other_output {
                // Ignore order edges
                continue;
            } else if Some(out_port) == static_output {
                let EdgeKind::Const(ty) = op.static_output().unwrap() else {
                    return Err(PytketEncodeError::custom(format!(
                        "Cannot emit a static output for a {op}."
                    )));
                };
                ty
            } else {
                let Some(ty) = signature
                    .as_ref()
                    .and_then(|s| s.out_port_type(out_port).cloned())
                else {
                    return Err(PytketEncodeError::custom(
                        "Cannot emit a transparent node without a dataflow signature.",
                    ));
                };
                ty
            };

            let wire = hugr::Wire::new(node, out_port);
            let Some(count) = self.config().type_to_pytket(&ty) else {
                return Err(PytketEncodeError::custom(format!(
                    "Found an unsupported type {ty} while encoding a {op}."
                )));
            };
            wire_counts.push((wire, count));
        }
        Ok(wire_counts)
    }
}

/// Result of trying to encode a node in the Hugr.
///
/// This flag indicates that either
/// - The operation was successful encoded and is now registered on the
///   [`PytketEncoderContext`]
/// - The node cannot be encoded, and the context was left unchanged.
///
/// The latter is a recoverable error, as it promises that the context wasn't
/// modified. For non-recoverable errors, see [`PytketEncodeError`].
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, derive_more::Display)]
pub enum EncodeStatus {
    /// The node was successfully encoded and registered in the context.
    Success,
    /// The node could not be encoded, and the context was left unchanged.
    Unsupported,
}

/// Input arguments to the output parameter computation methods in the the emit_*
/// functions of [`PytketEncoderContext`].
#[derive(Clone, Copy, Debug)]
pub struct OutputParamArgs<'a> {
    /// The expected number of output parameters.
    pub expected_count: usize,
    /// The list of input parameter expressions.
    pub input_params: &'a [String],
}

/// Input arguments to the output parameter computation method in
/// [`PytketEncoderContext::emit_node_command`].
///
/// This can be passed to [`make_tk1_operation`] to create a pytket
/// [`circuit_json::Operation`].
#[derive(Clone, Debug)]
pub struct MakeOperationArgs<'a> {
    /// Number of input qubits.
    pub num_qubits: usize,
    /// Number of input bits.
    pub num_bits: usize,
    /// List of input parameter expressions.
    pub params: Cow<'a, [String]>,
}

/// Tracked values in a node's inputs, and any remaining input wire with missing
/// value information.
///
/// In most cases, finding an unsupported wire should be an error (see
/// [`NodeInputValues::try_into_tracked_values`]).
///
/// Auxiliary struct returned by
/// [`PytketEncoderContext::get_input_values_internal`]
struct NodeInputValues<N> {
    /// Tracked values originating in the local region.
    pub tracked_values: TrackedValues,
    /// Untracked inputs, with unknown values.
    pub unknown_values: Vec<Wire<N>>,
}

impl<N: HugrNode> NodeInputValues<N> {
    /// Return the tracked values in the node's inputs.
    ///
    /// # Errors
    /// - [`OpConvertError::WireHasNoValues`] if there were any unknown wires.
    pub fn try_into_tracked_values(self) -> Result<TrackedValues, PytketEncodeError<N>> {
        match self.unknown_values.is_empty() {
            true => Ok(self.tracked_values),
            false => Err(OpConvertError::WireHasNoValues {
                wire: self.unknown_values[0],
            }
            .into()),
        }
    }
}

/// Cached value for a function encoding.
///
/// If the function contains output parameters, it is unsupported
/// and should be emitted as an unsupported op instead.
#[derive(Clone, Debug)]
enum CachedEncodedFunction {
    /// Successfully encoded function.
    Encoded {
        /// The serialised circuit for the function.
        serial_circuit: SerialCircuit,
    },
    /// Unsupported function
    Unsupported,
    /// A marker for functions currently being encoded.
    ///
    /// Used to detect recursive calls, and prevent infinite recursion.
    InEncodingStack,
}

/// Initialize a tket1 [Operation](circuit_json::Operation) to pass to
/// [`PytketEncoderContext::emit_command`].
///
/// ## Arguments
/// - `pytket_optype`: The operation type to use.
/// - `qubit_count`: The number of qubits used by the operation.
/// - `bit_count`: The number of linear bits used by the operation.
/// - `params`: Parameters of the operation, expressed as string expressions.
///   Normally obtained from [`ValueTracker::param_expression`].
pub fn make_tk1_operation(
    pytket_optype: tket_json_rs::OpType,
    inputs: MakeOperationArgs<'_>,
) -> circuit_json::Operation {
    let mut op = circuit_json::Operation::default();
    op.op_type = pytket_optype;
    op.n_qb = Some(inputs.num_qubits as u32);
    op.params = match inputs.params.is_empty() {
        false => Some(inputs.params.into_owned()),
        true => None,
    };
    op.signature = Some(
        [
            vec!["Q".into(); inputs.num_qubits],
            vec!["B".into(); inputs.num_bits],
        ]
        .concat(),
    );
    op
}

/// Initialize a tket1 [Operation](circuit_json::Operation) that only operates
/// on classical values.
///
/// This method is specific to some classical operations in
/// [`tket_json_rs::OpType`]. For the classical _expressions_ in
/// [`tket_json_rs::OpType::ClExpr`] / [`tket_json_rs::clexpr::op::ClOp`] use
/// [`make_tk1_classical_expression`].
///
/// This can be passed to [`PytketEncoderContext::emit_command`].
///
/// See [`make_tk1_operation`] for a more general case.
///
/// ## Arguments
/// - `pytket_optype`: The operation type to use.
/// - `bit_count`: The number of linear bits used by the operation.
/// - `classical`: The parameters to the classical operation.
pub fn make_tk1_classical_operation(
    pytket_optype: tket_json_rs::OpType,
    bit_count: usize,
    classical: tket_json_rs::circuit_json::Classical,
) -> tket_json_rs::circuit_json::Operation {
    let args = MakeOperationArgs {
        num_qubits: 0,
        num_bits: bit_count,
        params: Cow::Borrowed(&[]),
    };
    let mut op = make_tk1_operation(pytket_optype, args);
    op.classical = Some(Box::new(classical));
    op
}

/// Initialize a tket1 [Operation](circuit_json::Operation) that represents a
/// classical expression.
///
/// This method is specific to [`tket_json_rs::OpType::ClExpr`] /
/// [`tket_json_rs::clexpr::op::ClOp`]. For other classical operations in
/// [`tket_json_rs::OpType`] use [`make_tk1_classical_operation`].
///
/// Classical expressions are a bit different from other operations in pytket as
/// they refer to their inputs and outputs by their position in the operation's
/// bit and register list. See the arguments below for more details.
///
/// This can be passed to [`PytketEncoderContext::emit_command`]. See
/// [`make_tk1_operation`] for a more general case.
///
/// ## Arguments
/// - `bit_count`: The number of bits (both inputs and outputs) referenced by
///   the operation. The operation will refer to the bits by an index in the
///   range `0..bit_count`.
/// - `output_bits`: Slice of bit indices in `0..bit_count` that are the outputs
///   of the operation.
/// - `registers`: groups of bits that are used as registers in the operation.
///   Each group is a slice of bit indices in `0..bit_count`, plus a register
///   identifier. The operation may refer to these registers.
/// - `expression`: The classical expression, expressed in term of the local
///   bit and register indices.
pub fn make_tk1_classical_expression(
    bit_count: usize,
    output_bits: &[u32],
    registers: &[InputClRegister],
    expression: tket_json_rs::clexpr::operator::ClOperator,
) -> tket_json_rs::circuit_json::Operation {
    let mut clexpr = tket_json_rs::clexpr::ClExpr::default();
    clexpr.bit_posn = (0..bit_count as u32).map(|i| (i, i)).collect();
    clexpr.reg_posn = registers.to_vec();
    clexpr.output_posn = tket_json_rs::clexpr::ClRegisterBits(output_bits.to_vec());
    clexpr.expr = expression;

    let args = MakeOperationArgs {
        num_qubits: 0,
        num_bits: bit_count,
        params: Cow::Borrowed(&[]),
    };
    let mut op = make_tk1_operation(tket_json_rs::OpType::ClExpr, args);
    op.classical_expr = Some(clexpr);
    op
}
