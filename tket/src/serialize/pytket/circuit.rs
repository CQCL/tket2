//! Temporary structure linking an encoded pytket circuit and subcircuits, with their originating HUGR.

use std::collections::{HashMap, VecDeque};
use std::ops::{Index, IndexMut};

use hugr::core::HugrNode;
use hugr::ops::handle::NodeHandle;
use hugr::ops::{OpTag, OpTrait};
use hugr::{Hugr, HugrView, Node};
use rayon::iter::{IntoParallelIterator, IntoParallelRefIterator, IntoParallelRefMutIterator};
use tket_json_rs::circuit_json::{Command as PytketCommand, SerialCircuit};

use crate::serialize::pytket::decoder::PytketDecoderContext;
use crate::serialize::pytket::{
    default_encoder_config, DecodeInsertionTarget, DecodeOptions, EncodeOptions, PytketDecodeError,
    PytketDecodeErrorInner, PytketEncodeError, PytketEncoderContext,
};
use crate::Circuit;

use super::opaque::OpaqueSubgraphs;

/// An encoded pytket circuit that may be linked to an existing HUGR.
///
/// Tracks correspondences between references to the HUGR in the encoded
/// circuit, so we can reconstruct the HUGR if needed.
///
/// Serial circuits in this structure are intended to be transient, only alive
/// while this structure is in memory. To obtain a fully standalone pytket
/// circuit that can be used independently, and stored permanently, use
/// [`EncodedCircuit::new_standalone`] or call
/// [`EncodedCircuit::ensure_standalone`].
pub struct EncodedCircuit<Node: HugrNode> {
    /// Region in the HUGR that was encoded as the main circuit.
    ///
    /// If [`EncodeOptions::encode_subcircuits`] was set during the encoding
    /// process, `circuits` will contain entries for some dataflow regions that
    /// descendants of this node.
    ///
    /// If [`EncodeOptions::encode_subcircuits`] was not set, `circuits` will
    /// only contain an entry for this region if it was a dataflow container, or
    /// no entries if it was not.
    head_region: Node,
    /// Circuits encoded from independent dataflow regions in the HUGR.
    ///
    /// These correspond to sections of the HUGR that can be optimized
    /// independently.
    circuits: HashMap<Node, SerialCircuit>,
    /// Sets of subgraphs in the HUGR that have been encoded as opaque barriers
    /// in the pytket circuit.
    ///
    /// Subcircuits are identified in the barrier metadata by their ID in this
    /// vector. See [`SubgraphId`].
    opaque_subgraphs: OpaqueSubgraphs<Node>,
}

impl EncodedCircuit<Node> {
    /// Encode a HugrView into a [`EncodedCircuit`].
    ///
    /// The HUGR's entrypoint must be a dataflow region that will be encoded as
    /// the main circuit. Additional circuits may be encoded if
    /// [`EncodeOptions::encode_subcircuits`] is set.
    ///
    /// The circuit may contain opaque barriers referencing subgraphs in the
    /// original HUGR. To obtain a fully standalone pytket circuit that can be
    /// used independently, and stored permanently, use
    /// [`EncodedCircuit::new_standalone`] or call
    /// [`EncodedCircuit::ensure_standalone`].
    ///
    /// See [`EncodeOptions`] for the options used by the encoder.
    pub fn new<H: AsRef<Hugr> + AsMut<Hugr> + HugrView<Node = Node>>(
        circuit: &Circuit<H>,
        options: EncodeOptions<H>,
    ) -> Result<Self, PytketEncodeError<H::Node>> {
        let mut enc = Self {
            head_region: circuit.parent(),
            circuits: HashMap::new(),
            opaque_subgraphs: OpaqueSubgraphs::new(0),
        };

        enc.encode_circuits(circuit, options)?;
        enc.ensure_standalone(circuit.hugr())?;

        Ok(enc)
    }

    /// Reassemble the encoded circuits inside an existing [`Hugr`], containing
    /// the [`Self::head_region`] at the given insertion target.
    ///
    /// Functions called by the internal hugrs may be added to the hugr module
    /// as well.
    ///
    /// # Arguments
    ///
    /// - `hugr`: The [`Hugr`] to reassemble the circuits in.
    /// - `target`: The target to insert the function at.
    /// - `options`: The options for the decoder.
    ///
    /// # Errors
    ///
    /// Returns a [`PytketDecodeErrorInner::NonDataflowHeadRegion`] error if
    /// [`Self::head_region`] is not a dataflow container in the hugr.
    ///
    /// Returns an error if a circuit being decoded is invalid. See
    /// [`PytketDecodeErrorInner`][super::error::PytketDecodeErrorInner] for
    /// more details.
    pub fn reassemble_inline<H: AsRef<Hugr> + AsMut<Hugr> + HugrView<Node = Node>>(
        &self,
        hugr: &mut Hugr,
        target: DecodeInsertionTarget,
        options: DecodeOptions,
    ) -> Result<hugr::Node, PytketDecodeError> {
        self.check_dataflow_head_region(self.head_region)
            .map_err(|_| {
                PytketDecodeErrorInner::NonDataflowHeadRegion {
                    head_op: Some(hugr.get_optype(self.head_region).to_string()),
                }
                .wrap()
            })?;
        let serial_circuit = &self[self.head_region];

        if self.len() > 1 {
            unimplemented!(
                "Reassembling an `EncodedCircuit` with nested subcircuits is not yet implemented."
            );
        };

        let mut decoder = PytketDecoderContext::new(serial_circuit, hugr, target, options)?;
        decoder.register_opaque_subgraphs(&self.opaque_subgraphs);
        decoder.run_decoder(&serial_circuit.commands)?;
        Ok(decoder.finish()?.node())
    }
}

impl<Node: HugrNode> EncodedCircuit<Node> {
    /// Encode a HugrView into a [`EncodedCircuit`].
    ///
    /// The HUGR's entrypoint must be a dataflow region that will be encoded as
    /// the main circuit. Additional circuits may be encoded if
    /// [`EncodeOptions::encode_subcircuits`] is set.
    ///
    /// The circuit may contain opaque barriers encoding opaque subgraphs in the
    /// original HUGR. These are encoded completely as Hugr envelopes in the
    /// barrier operations metadata.
    ///
    /// When encoding a `Hugr`, prefer using [`EncodedCircuit::new`] instead to
    /// avoid unnecessary copying of the opaque subgraphs.
    ///
    /// See [`EncodeOptions`] for the options used by the encoder.
    pub fn new_standalone<H: HugrView<Node = Node>>(
        circuit: &Circuit<H>,
        options: EncodeOptions<H>,
    ) -> Result<Self, PytketEncodeError<H::Node>> {
        let mut enc = Self {
            head_region: circuit.parent(),
            circuits: HashMap::new(),
            opaque_subgraphs: OpaqueSubgraphs::new(0),
        };

        enc.encode_circuits(circuit, options)?;

        Ok(enc)
    }

    /// Encode the circuits for the entrypoint region to the hugr, and if [`EncodeOptions::encode_subcircuits`] is set,
    /// for the descendants of any unsupported node in the main circuit.
    ///
    /// Auxiliary method for [`Self::from_hugr`].
    ///
    /// TODO: Add an option in [EncodeOptions] to run the subcircuit encoders in parallel.
    fn encode_circuits<H: HugrView<Node = Node>>(
        &mut self,
        // This is already in [`self.hugr`], but we pass it since wrapping it
        // again results in a `Circuit<&H>`, which doesn't play well with
        // `config`.
        circuit: &Circuit<H>,
        mut options: EncodeOptions<H>,
    ) -> Result<(), PytketEncodeError<H::Node>> {
        // List of nodes to check for subcircuits.
        //
        // These may be either dataflow region parents that we can encode, or
        // any node with children that we should traverse recursively until we
        // find a dataflow region.
        let mut candidate_nodes = VecDeque::from([self.head_region]);
        let config = match options.config.take() {
            Some(config) => config,
            None => default_encoder_config().into(),
        };

        // Add a node to the list of candidates if it's a region parent.
        let add_candidate = |node: H::Node, queue: &mut VecDeque<H::Node>| {
            if circuit.hugr().first_child(node).is_some() {
                queue.push_back(node);
            }
        };

        // Add all container nodes from the new opaque subgraphs to the list of
        // candidates.
        let add_subgraph_candidates =
            |subgraphs: &OpaqueSubgraphs<H::Node>, queue: &mut VecDeque<H::Node>| {
                for subgraph_id in subgraphs.ids() {
                    for &node in subgraphs[subgraph_id].nodes() {
                        add_candidate(node, queue);
                    }
                }
            };

        let mut encoder_count = 0;
        while let Some(node) = candidate_nodes.pop_front() {
            let node_op = circuit.hugr().get_optype(node);
            if !OpTag::DataflowParent.is_superset(node_op.tag()) {
                for child in circuit.hugr().children(node) {
                    add_candidate(child, &mut candidate_nodes);
                }
                continue;
            }
            encoder_count += 1;
            let opaque_subgraphs = OpaqueSubgraphs::new(encoder_count);
            let mut encoder: PytketEncoderContext<H> =
                PytketEncoderContext::new(circuit, node, opaque_subgraphs, config.clone())?;
            encoder.run_encoder(circuit, node)?;
            let (serial, _, opaque_subgraphs) = encoder.finish(circuit, node)?;

            if options.encode_subcircuits {
                add_subgraph_candidates(&opaque_subgraphs, &mut candidate_nodes);
            }

            self.circuits.insert(node, serial);
            self.opaque_subgraphs.merge(opaque_subgraphs);
        }

        Ok(())
    }

    /// Reassemble the encoded circuits into a new [`Hugr`], containing a
    /// function defining the [`Self::head_region`] and expanding any opaque
    /// hugrs in pytket barrier operations back into Hugr subgraphs.
    ///
    /// Functions called by the internal hugrs may be added to the hugr module
    /// as well.
    ///
    /// # Arguments
    ///
    /// - `fn_name`: The name of the function to create. If `None`, we will use
    ///   the name of the circuit, or "main" if the circuit has no name.
    /// - `options`: The options for the decoder.
    ///
    /// # Errors
    ///
    /// Returns a [`PytketDecodeErrorInner::NonDataflowHeadRegion`] error if
    /// [`Self::head_region`] is not a dataflow container in the hugr.
    ///
    /// Returns an error if a circuit being decoded is invalid. See
    /// [`PytketDecodeErrorInner`][super::error::PytketDecodeErrorInner] for
    /// more details.
    pub fn reassemble(
        &self,
        fn_name: Option<String>,
        options: DecodeOptions,
    ) -> Result<Hugr, PytketDecodeError> {
        self.check_dataflow_head_region(self.head_region)?;
        let serial_circuit = &self[self.head_region];

        if self.len() > 1 {
            unimplemented!(
                "Reassembling an `EncodedCircuit` with nested subcircuits is not yet implemented."
            );
        };

        let mut hugr = Hugr::new();
        let target = DecodeInsertionTarget::Function { fn_name };

        let mut decoder = PytketDecoderContext::new(serial_circuit, &mut hugr, target, options)?;
        decoder.run_decoder(&serial_circuit.commands)?;
        decoder.finish()?;
        Ok(hugr)
    }

    /// Ensure that none of the encoded circuits contain references to opaque subgraphs in the original HUGR.
    ///
    /// Traverses the commands in the encoded circuits and replaces
    /// [`OpaqueSubgraphPayloadType::External`][super::opaque::OpaqueSubgraphPayloadType::External]
    /// pointers in opaque barriers with inline payloads.
    ///
    /// # Errors
    ///
    /// Returns an error if a barrier operation with the
    /// [`OPGROUP_OPAQUE_HUGR`][super::opaque::OPGROUP_OPAQUE_HUGR]
    /// opgroup has an invalid payload.
    pub fn ensure_standalone(
        &mut self,
        hugr: &impl HugrView<Node = Node>,
    ) -> Result<(), PytketEncodeError<Node>> {
        /// Replace references to the `EncodedCircuit` context from the circuit commands.
        ///
        /// Replaces [`OpaqueSubgraphPayloadType::External`][super::opaque::OpaqueSubgraphPayloadType::External]
        /// pointers in opaque barriers with inline payloads.
        fn make_commands_standalone<N: HugrNode>(
            commands: &mut [PytketCommand],
            subgraphs: &OpaqueSubgraphs<N>,
            hugr: &impl HugrView<Node = N>,
        ) -> Result<(), PytketEncodeError<N>> {
            for command in commands.iter_mut() {
                subgraphs.inline_payload(command, hugr)?;

                if let Some(tket_json_rs::opbox::OpBox::CircBox { circuit, .. }) =
                    &mut command.op.op_box
                {
                    make_commands_standalone(&mut circuit.commands, subgraphs, hugr)?;
                }
            }
            Ok(())
        }

        for serial_circuit in self.circuits.values_mut() {
            make_commands_standalone(&mut serial_circuit.commands, &self.opaque_subgraphs, hugr)?;
        }
        Ok(())
    }

    /// Checks if [`Self::head_region`] was a dataflow container in the original hugr,
    /// and therefore has an encoded circuit in this structure.
    ///
    /// # Errors
    ///
    /// Returns a [`PytketEncodeError::InvalidStandaloneHeadRegion`] error if
    /// [`Self::head_region`] is not a dataflow container encoded from the original hugr.
    ///
    /// If `hugr` is provided, the error will include the operation type of the head region.
    fn check_dataflow_head_region(&self, head_region: Node) -> Result<(), PytketDecodeError> {
        if !self.circuits.contains_key(&head_region) {
            return Err(PytketDecodeErrorInner::NonDataflowHeadRegion { head_op: None }.wrap());
        }
        Ok(())
    }

    /// Returns the region node from which the main circuit was encoded.
    pub fn head_region(&self) -> Node {
        self.head_region
    }

    /// Returns an iterator over all the encoded pytket circuits.
    pub fn circuits(&self) -> impl Iterator<Item = (Node, &SerialCircuit)> {
        self.into_iter().map(|(&n, circ)| (n, circ))
    }

    /// Returns an iterator over all the encoded pytket circuits as mutable
    /// references.
    ///
    /// The circuits may be modified arbitrarily, as long as
    /// [`OpaqueSubgraphPayloadType::External`][super::opaque::OpaqueSubgraphPayloadType::External]
    /// pointers to HUGR subgraphs in opaque barriers remain valid and
    /// topologically consistent with the original circuit.
    pub fn circuits_mut(&mut self) -> impl Iterator<Item = (Node, &mut SerialCircuit)> {
        self.into_iter().map(|(&n, circ)| (n, circ))
    }

    /// Returns `true` if there is an encoded pytket circuit for the given region.
    pub fn contains_circuit(&self, region: Node) -> bool {
        self.circuits.contains_key(&region)
    }

    /// Returns the number of encoded pytket circuits.
    pub fn len(&self) -> usize {
        self.circuits.len()
    }

    /// Returns whether the encoded circuit is empty.
    pub fn is_empty(&self) -> bool {
        self.circuits.is_empty()
    }
}

impl<Node: HugrNode> Index<Node> for EncodedCircuit<Node> {
    type Output = SerialCircuit;

    fn index(&self, index: Node) -> &Self::Output {
        &self.circuits[&index]
    }
}

impl<Node: HugrNode> IndexMut<Node> for EncodedCircuit<Node> {
    fn index_mut(&mut self, index: Node) -> &mut Self::Output {
        self.circuits
            .get_mut(&index)
            .unwrap_or_else(|| panic!("Indexing into a circuit that was not encoded: {index}"))
    }
}

impl<'c, Node: HugrNode> IntoIterator for &'c EncodedCircuit<Node> {
    type Item = (&'c Node, &'c SerialCircuit);
    type IntoIter = <&'c HashMap<Node, SerialCircuit> as IntoIterator>::IntoIter;

    fn into_iter(self) -> Self::IntoIter {
        self.circuits.iter()
    }
}

impl<'c, Node: HugrNode> IntoIterator for &'c mut EncodedCircuit<Node> {
    type Item = (&'c Node, &'c mut SerialCircuit);
    type IntoIter = <&'c mut HashMap<Node, SerialCircuit> as IntoIterator>::IntoIter;

    fn into_iter(self) -> Self::IntoIter {
        self.circuits.iter_mut()
    }
}

impl<'c, Node: HugrNode> IntoParallelIterator for &'c EncodedCircuit<Node>
where
    Node: Send + Sync,
{
    type Item = (&'c Node, &'c SerialCircuit);
    type Iter = <&'c HashMap<Node, SerialCircuit> as IntoParallelIterator>::Iter;

    fn into_par_iter(self) -> Self::Iter {
        self.circuits.par_iter()
    }
}

impl<'c, Node: HugrNode> IntoParallelIterator for &'c mut EncodedCircuit<Node>
where
    Node: Send + Sync,
{
    type Item = (&'c Node, &'c mut SerialCircuit);
    type Iter = <&'c mut HashMap<Node, SerialCircuit> as IntoParallelIterator>::Iter;

    fn into_par_iter(self) -> Self::Iter {
        self.circuits.par_iter_mut()
    }
}
