//! Utility

use std::collections::HashMap;

use hugr::builder::{Dataflow, DataflowHugr, FunctionBuilder};
use hugr::extension::ExtensionSet;
use hugr::hugr::hugrmut::HugrMut;
use hugr::hugr::views::sibling_subgraph::ConvexChecker;
use hugr::hugr::views::{HierarchyView, SiblingGraph, SiblingSubgraph};
use hugr::hugr::NodeMetadata;
use hugr::ops::handle::FuncID;
use hugr::types::{FunctionType, Signature};
use hugr::{Hugr, HugrView, Node, Port, Wire};
use itertools::Itertools;

use crate::extension::REGISTRY;
use crate::Circuit;

/// A chunk of a circuit.
#[derive(Debug, Clone)]
pub struct Chunk {
    /// The extracted circuit.
    pub circ: Hugr,
    /// The original wires connected to the input.
    pub inputs: Vec<Wire>,
    /// The original wires connected to the output.
    pub outputs: Vec<Wire>,
}

/*
TODO: Replace the Wires used in this module with better identifiers.

/// An identifier for the source of an inter-chunk edge.
///
/// This is required for re-connecting the chunks during reassembly.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ChunkEdgeSource {
    /// The edge was connected to the input of the original circuit.
    Input(Port),
    /// The edge was connected to another chunk.
    ///
    /// The source is identified by its node and port in the original circuit.
    ChunkOutput(Node, Port),
}
*/

impl Chunk {
    /// Extract a chunk from a circuit.
    ///
    /// The chunk is extracted from the input wires to the output wires.
    pub(self) fn extract<'h, H: HugrView>(
        circ: &'h H,
        nodes: impl IntoIterator<Item = Node>,
        checker: &mut ConvexChecker<'h, H>,
    ) -> Self {
        let subgraph = SiblingSubgraph::try_from_nodes_with_checker(
            nodes.into_iter().collect_vec(),
            circ,
            checker,
        )
        .expect("Failed to define the chunk subgraph");
        let extracted = subgraph
            .extract_subgraph(circ, "Chunk", ExtensionSet::new())
            .expect("Failed to extract chunk");
        // Transform the subgraph's input/output sets into wires that can be
        // matched between different chunks.
        //
        // This requires finding the `Outgoing` port corresponding to each
        // subgraph input.
        let inputs = subgraph
            .incoming_ports()
            .iter()
            .map(|wires| {
                let (inp_node, inp_port) = wires[0];
                let (out_node, out_port) =
                    circ.linked_ports(inp_node, inp_port).exactly_one().ok().unwrap();
                Wire::new(out_node, out_port)
            })
            .collect();
        let outputs = subgraph
            .outgoing_ports()
            .iter()
            .map(|&(node, port)| Wire::new(node, port))
            .collect();
        Self {
            circ: extracted,
            inputs,
            outputs,
        }
    }

    /// Insert the chunk back into a circuit.
    ///
    /// Returns a map from the original input/output wires to the inserted [`IncomingPorts`]/[`OutgoingPorts`].
    //
    // TODO: The new chunk may have input ports directly connected to outputs. We have to take care of those.
    pub(self) fn insert(
        &self,
        circ: &mut impl HugrMut,
        root: Node,
    ) -> (
        HashMap<Wire, Vec<(Node, Port)>>,
        HashMap<Wire, (Node, Port)>,
    ) {
        let chunk_sg: SiblingGraph<'_, FuncID<true>> =
            SiblingGraph::try_new(&self.circ, self.circ.root()).unwrap();
        let subgraph = SiblingSubgraph::try_new_dataflow_subgraph(&chunk_sg)
            .expect("The chunk circuit is no longer a dataflow");
        let node_map = circ
            .insert_subgraph(root, &self.circ, &subgraph)
            .expect("Failed to insert the chunk subgraph")
            .node_map;

        let [inp, out] = circ.get_io(root).unwrap();
        let mut input_map = HashMap::with_capacity(self.inputs.len());
        let mut output_map = HashMap::with_capacity(self.outputs.len());

        for (&wire, incoming) in self.inputs.iter().zip(subgraph.incoming_ports().iter()) {
            let incoming = incoming.iter().map(|&(node, port)| {
                if node == out {
                    // TODO: Add a map for directly connected Input Wire -> Output Wire.
                    panic!("Chunk input directly connected to the output. This is not currently supported.");
                }
                (*node_map.get(&node).unwrap(),port)
            }).collect_vec();
            input_map.insert(wire, incoming);
        }

        for (&wire, &(node, port)) in self.outputs.iter().zip(subgraph.outgoing_ports().iter()) {
            if node == inp {
                // TODO: Add a map for directly connected Input Wire -> Output Wire.
                panic!("Chunk input directly connected to the output. This is not currently supported.");
            }
            output_map.insert(wire, (*node_map.get(&node).unwrap(), port));
        }

        (input_map, output_map)
    }
}

/// An utility for splitting a circuit into chunks, and reassembling them afterwards.
#[derive(Debug, Clone)]
pub struct CircuitChunks {
    /// The original circuit's signature.
    signature: FunctionType,

    /// The original circuit's root metadata.
    root_meta: NodeMetadata,

    /// The original circuit's input node. Required to identify chunk edges that
    /// where connected to the input.
    circ_input: Node,

    /// The original circuit's output node. Required to identify chunk edges
    /// that where connected to the output.
    circ_output: Node,

    /// The split circuits.
    pub chunks: Vec<Chunk>,
}

impl CircuitChunks {
    /// Split a circuit into chunks.
    ///
    /// The circuit is split into chunks of at most `max_size` gates.
    pub fn split(circ: &impl Circuit, max_size: usize) -> Self {
        let root_meta = circ.get_metadata(circ.root()).clone();
        let signature = circ.circuit_signature().clone();
        let [circ_input, circ_output] = circ.get_io(circ.root()).unwrap();

        let mut chunks = Vec::new();
        let mut convex_checker = ConvexChecker::new(circ);
        for commands in &circ.commands().chunks(max_size) {
            chunks.push(Chunk::extract(
                circ,
                commands.map(|cmd| cmd.node()),
                &mut convex_checker,
            ));
        }
        Self {
            signature,
            root_meta,
            circ_input,
            circ_output,
            chunks,
        }
    }

    /// Reassemble the chunks into a circuit.
    pub fn reassemble(self) -> Result<Hugr, ()> {
        let name = self
            .root_meta
            .get("name")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        let signature = Signature {
            signature: self.signature,
            // TODO: Is this correct? Can a circuit root have a fixed set of input extensions?
            input_extensions: ExtensionSet::new(),
        };

        let builder = FunctionBuilder::new(name, signature).unwrap();
        let inputs = builder.input_wires();
        let mut reassembled = builder.finish_hugr_with_outputs(inputs, &REGISTRY).unwrap();
        let root = reassembled.root();
        let [reassembled_input, reassembled_output] = reassembled.get_io(root).unwrap();

        // The chunks input and outputs are each identified with a wire in the original circuit.
        // We collect both sides first, and rewire them after the chunks have been inserted.
        let mut sources: HashMap<Wire, (Node, Port)> = HashMap::new();
        let mut targets: HashMap<Wire, Vec<(Node, Port)>> = HashMap::new();

        for chunk in self.chunks {
            // Insert the chunk circuit without its input/output nodes.
            let (chunk_targets, chunk_sources) = chunk.insert(&mut reassembled, root);
            // Reconnect the chunk's inputs and outputs in the reassembled circuit.
            sources.extend(chunk_sources);
            chunk_targets.into_iter().for_each(|(wire, tgts)| {
                targets.entry(wire).or_default().extend(tgts);
            });
        }

        // Reconnect the different chunks.
        /* TODO
        for (wire, source) in sources {
            for (target, port) in targets.remove(&wire).unwrap() {
                reassembled.link(source.0, source.1, target, port);
            }
        }
        */
        let _ = (
            reassembled_input,
            reassembled_output,
            self.circ_input,
            self.circ_output,
        );

        Ok(reassembled)
    }
}

#[cfg(test)]
mod test {
    use crate::circuit::CircuitHash;
    use crate::utils::build_simple_circuit;
    use crate::T2Op;

    use super::*;

    #[test]
    fn split_reassemble() {
        let circ = build_simple_circuit(2, |circ| {
            circ.append(T2Op::H, [0])?;
            circ.append(T2Op::CX, [0, 1])?;
            circ.append(T2Op::T, [1])?;
            circ.append(T2Op::H, [0])?;
            circ.append(T2Op::CX, [0, 1])?;
            circ.append(T2Op::H, [0])?;
            circ.append(T2Op::CX, [0, 1])?;
            Ok(())
        })
        .unwrap();

        crate::utils::test::viz_dotstr(&circ.dot_string());
        let chunks = CircuitChunks::split(&circ, 3);
        let reassembled = chunks.reassemble().unwrap();
        crate::utils::test::viz_dotstr(&reassembled.dot_string());

        assert_eq!(circ.circuit_hash(), reassembled.circuit_hash());
    }
}
