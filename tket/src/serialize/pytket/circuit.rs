//! Temporary structure linking an encoded pytket circuit and subcircuits, with their originating HUGR.

use std::collections::{HashMap, VecDeque};
use std::ops::{Index, IndexMut};

use hugr::core::HugrNode;
use hugr::ops::{OpTag, OpTrait};
use hugr::{Hugr, HugrView};
use rayon::iter::{IntoParallelIterator, IntoParallelRefIterator, IntoParallelRefMutIterator};
use tket_json_rs::circuit_json::{Command as PytketCommand, SerialCircuit};

use crate::serialize::pytket::{
    default_encoder_config, EncodeOptions, PytketEncodeError, PytketEncoderContext,
};
use crate::Circuit;

use super::unsupported::UnsupportedSubgraphs;

/// An encoded pytket circuit that may be linked to an existing HUGR.
///
/// Tracks correspondences between references to the HUGR in the encoded
/// circuit, so we can reconstruct the HUGR if needed.
///
/// Serial circuits in this structure are intended to be transient, only alive
/// while this structure is in memory.
/// To obtain a fully standalone pytket circuit that can be used independently,
/// and stored permanently, use [`EncodedCircuit::extract_standalone`].
pub struct EncodedCircuit<'a, H: HugrView = Hugr> {
    /// Region in the HUGR that was encoded as the main circuit.
    ///
    /// If [`EncodeOptions::encode_subcircuits`] was set during the encoding
    /// process, `circuits` will contain entries for some dataflow regions that
    /// descendants of this node.
    ///
    /// If [`EncodeOptions::encode_subcircuits`] was not set, `circuits` will
    /// only contain an entry for this region if it was a dataflow container, or
    /// no entries if it was not.
    head_region: H::Node,
    /// Circuits encoded from dataflow regions in the HUGR.
    ///
    /// These correspond to disjoint sections of the HUGR and can be optimized
    /// independently.
    circuits: HashMap<H::Node, SerialCircuit>,
    /// Sets of subgraphs in the HUGR that have been encoded as opaque barriers
    /// in the pytket circuit.
    ///
    /// Subcircuits are identified in the barrier metadata by their ID in this
    /// vector. See [`SubgraphId`].
    opaque_subgraphs: UnsupportedSubgraphs<H::Node>,
    /// The HUGR from where the pytket circuits were encoded.
    hugr: &'a H,
}

impl<'a, H: HugrView> EncodedCircuit<'a, H> {
    /// Encode a Hugr into a [`EncodedCircuit`].
    ///
    /// The HUGR's entrypoint must be a dataflow region that will be encoded as
    /// the main circuit. Additional circuits may be encoded if
    /// [`EncodeOptions::encode_subcircuits`] is set.
    ///
    /// The circuit may contain opaque barriers referencing subgraphs in the
    /// original HUGR. To extract a fully standalone pytket circuit that can be
    /// used independently, use [`EncodedCircuit::extract_standalone`].
    ///
    /// See [`EncodeOptions`] for the options used by the encoder.
    pub fn from_hugr(
        circuit: &'a Circuit<H>,
        options: EncodeOptions<H>,
    ) -> Result<Self, PytketEncodeError<H::Node>> {
        let mut enc = Self {
            head_region: circuit.parent(),
            circuits: HashMap::new(),
            opaque_subgraphs: UnsupportedSubgraphs::new(),
            hugr: circuit.hugr(),
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
    fn encode_circuits(
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
            |subgraphs: &UnsupportedSubgraphs<H::Node>, queue: &mut VecDeque<H::Node>| {
                for subgraph_id in subgraphs.ids() {
                    for &node in subgraphs.get_unsupported_subgraph(subgraph_id).nodes() {
                        add_candidate(node, queue);
                    }
                }
            };

        while let Some(node) = candidate_nodes.pop_front() {
            let node_op = circuit.hugr().get_optype(node);
            if !OpTag::DataflowParent.is_superset(node_op.tag()) {
                for child in circuit.hugr().children(node) {
                    add_candidate(child, &mut candidate_nodes);
                }
                continue;
            }
            let mut encoder: PytketEncoderContext<H> =
                PytketEncoderContext::new(circuit, node, config.clone())?;
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

    /// Extract the top-level pytket circuit as a standalone definition
    /// containing the whole original HUGR.
    ///
    /// Traverses the commands in `head_circuit` and replaces
    /// [`UnsupportedSubgraphPayloadType::External`][super::unsupported::UnsupportedSubgraphPayloadType::External]
    /// pointers in opaque barriers with standalone payloads.
    ///
    /// Discards any changes to the internal subcircuits, as they are not part
    /// of the top-level circuit.
    ///
    /// # Errors
    ///
    /// Returns a [`PytketEncodeError::InvalidStandaloneHeadRegion`] error if
    /// [`Self::head_region`] is not a dataflow container in the hugr.
    ///
    /// Returns an error if a barrier operation with the
    /// [`OPGROUP_EXTERNAL_UNSUPPORTED_HUGR`][super::unsupported::OPGROUP_EXTERNAL_UNSUPPORTED_HUGR]
    /// opgroup has an invalid payload.
    //
    // TODO: We'll need to handle non-local edges and function definitions in this step.
    pub fn extract_standalone(mut self) -> Result<SerialCircuit, PytketEncodeError<H::Node>> {
        if !self.check_dataflow_head_region() {
            let head_op = self.hugr.get_optype(self.head_region).to_string();
            return Err(PytketEncodeError::InvalidStandaloneHeadRegion { head_op });
        };
        let mut serial_circuit = self.circuits.remove(&self.head_region).unwrap();

        fn make_commands_standalone<N: HugrNode>(
            commands: &mut [PytketCommand],
            subgraphs: &UnsupportedSubgraphs<N>,
            hugr: &impl HugrView<Node = N>,
        ) -> Result<(), PytketEncodeError<N>> {
            for command in commands.iter_mut() {
                subgraphs.replace_external_with_standalone(command, hugr)?;

                if let Some(tket_json_rs::opbox::OpBox::CircBox { circuit, .. }) =
                    &mut command.op.op_box
                {
                    make_commands_standalone(&mut circuit.commands, subgraphs, hugr)?;
                }
            }
            Ok(())
        }
        make_commands_standalone(
            &mut serial_circuit.commands,
            &self.opaque_subgraphs,
            self.hugr,
        )?;

        Ok(serial_circuit)
    }

    /// Checks if [`Self::head_region`] was a dataflow container in the original hugr,
    /// and therefore has an encoded circuit in this structure.
    fn check_dataflow_head_region(&self) -> bool {
        self.circuits.contains_key(&self.head_region)
    }

    /// Returns the HUGR from where the circuit was encoded.
    pub fn hugr(&self) -> &H {
        self.hugr
    }

    /// Returns the region node from which the main circuit was encoded.
    pub fn head_region(&self) -> H::Node {
        self.head_region
    }

    /// Returns an iterator over all the encoded pytket circuits.
    pub fn circuits(&self) -> impl Iterator<Item = (H::Node, &SerialCircuit)> {
        self.into_iter().map(|(&n, circ)| (n, circ))
    }

    /// Returns an iterator over all the encoded pytket circuits as mutable
    /// references.
    ///
    /// The circuits may be modified arbitrarily, as long as
    /// [`UnsupportedSubgraphPayloadType::External`][super::unsupported::UnsupportedSubgraphPayloadType::External]
    /// pointers to HUGR subgraphs in opaque barriers remain valid and
    /// topologically consistent with the original circuit.
    pub fn circuits_mut(&mut self) -> impl Iterator<Item = (H::Node, &mut SerialCircuit)> {
        self.into_iter().map(|(&n, circ)| (n, circ))
    }

    /// Returns `true` if there is an encoded pytket circuit for the given region.
    pub fn contains(&self, region: H::Node) -> bool {
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

impl<'a, H: HugrView> Index<H::Node> for EncodedCircuit<'a, H> {
    type Output = SerialCircuit;

    fn index(&self, index: H::Node) -> &Self::Output {
        &self.circuits[&index]
    }
}

impl<'a, H: HugrView> IndexMut<H::Node> for EncodedCircuit<'a, H> {
    fn index_mut(&mut self, index: H::Node) -> &mut Self::Output {
        self.circuits
            .get_mut(&index)
            .unwrap_or_else(|| panic!("Indexing into a circuit that was not encoded: {index}"))
    }
}

impl<'c, 'a, H: HugrView> IntoIterator for &'c EncodedCircuit<'a, H> {
    type Item = (&'c H::Node, &'c SerialCircuit);
    type IntoIter = <&'c HashMap<H::Node, SerialCircuit> as IntoIterator>::IntoIter;

    fn into_iter(self) -> Self::IntoIter {
        self.circuits.iter()
    }
}

impl<'c, 'a, H: HugrView> IntoIterator for &'c mut EncodedCircuit<'a, H> {
    type Item = (&'c H::Node, &'c mut SerialCircuit);
    type IntoIter = <&'c mut HashMap<H::Node, SerialCircuit> as IntoIterator>::IntoIter;

    fn into_iter(self) -> Self::IntoIter {
        self.circuits.iter_mut()
    }
}

impl<'c, 'a, H> IntoParallelIterator for &'c EncodedCircuit<'a, H>
where
    H: HugrView,
    H::Node: Send + Sync,
{
    type Item = (&'c H::Node, &'c SerialCircuit);
    type Iter = <&'c HashMap<H::Node, SerialCircuit> as IntoParallelIterator>::Iter;

    fn into_par_iter(self) -> Self::Iter {
        self.circuits.par_iter()
    }
}

impl<'c, 'a, H> IntoParallelIterator for &'c mut EncodedCircuit<'a, H>
where
    H: HugrView,
    H::Node: Send + Sync,
{
    type Item = (&'c H::Node, &'c mut SerialCircuit);
    type Iter = <&'c mut HashMap<H::Node, SerialCircuit> as IntoParallelIterator>::Iter;

    fn into_par_iter(self) -> Self::Iter {
        self.circuits.par_iter_mut()
    }
}
