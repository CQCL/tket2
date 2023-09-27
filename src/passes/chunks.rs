//! Utility

use std::collections::HashMap;

use hugr::builder::{Dataflow, DataflowHugr, FunctionBuilder};
use hugr::extension::ExtensionSet;
use hugr::hugr::hugrmut::HugrMut;
use hugr::hugr::views::sibling_subgraph::ConvexChecker;
use hugr::hugr::views::{HierarchyView, SiblingGraph, SiblingSubgraph};
use hugr::hugr::{HugrError, NodeMetadata};
use hugr::ops::handle::DataflowParentID;
use hugr::types::{FunctionType, Signature};
use hugr::{Hugr, HugrView, Node, Port, Wire};
use itertools::Itertools;

use crate::extension::REGISTRY;
use crate::Circuit;

#[cfg(feature = "pyo3")]
use crate::json::TKETDecode;
#[cfg(feature = "pyo3")]
use pyo3::{exceptions::PyAttributeError, pyclass, pymethods, Py, PyAny, PyResult};
#[cfg(feature = "pyo3")]
use tket_json_rs::circuit_json::SerialCircuit;

/// An identifier for the connection between chunks.
///
/// This is based on the wires of the original circuit.
///
/// When reassembling the circuit, the input/output wires of each chunk are
/// re-linked by matching these identifiers.
pub type ChunkConnection = Wire;

/// A chunk of a circuit.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "pyo3", pyclass)]
pub struct Chunk {
    /// The extracted circuit.
    pub circ: Hugr,
    /// The original wires connected to the input.
    pub inputs: Vec<ChunkConnection>,
    /// The original wires connected to the output.
    pub outputs: Vec<ChunkConnection>,
}

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
                let (out_node, out_port) = circ
                    .linked_ports(inp_node, inp_port)
                    .exactly_one()
                    .ok()
                    .unwrap();
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
    #[allow(clippy::type_complexity)]
    pub(self) fn insert(
        &self,
        circ: &mut impl HugrMut,
        root: Node,
    ) -> (
        HashMap<ChunkConnection, Vec<(Node, Port)>>,
        HashMap<ChunkConnection, (Node, Port)>,
    ) {
        let chunk_sg: SiblingGraph<'_, DataflowParentID> =
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

        for (&connection, incoming) in self.inputs.iter().zip(subgraph.incoming_ports().iter()) {
            let incoming = incoming.iter().map(|&(node, port)| {
                if node == out {
                    // TODO: Add a map for directly connected Input connection -> Output Wire.
                    panic!("Chunk input directly connected to the output. This is not currently supported.");
                }
                (*node_map.get(&node).unwrap(),port)
            }).collect_vec();
            input_map.insert(connection, incoming);
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
#[cfg_attr(feature = "pyo3", pyclass)]
pub struct CircuitChunks {
    /// The original circuit's signature.
    signature: FunctionType,

    /// The original circuit's root metadata.
    root_meta: NodeMetadata,

    /// The original circuit's inputs.
    input_connections: Vec<ChunkConnection>,

    /// The original circuit's outputs.
    output_connections: Vec<ChunkConnection>,

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
        let input_connections = circ
            .node_outputs(circ_input)
            .map(|port| Wire::new(circ_input, port))
            .collect();
        let output_connections = circ
            .node_inputs(circ_output)
            .flat_map(|p| circ.linked_ports(circ_output, p))
            .map(|(n, p)| Wire::new(n, p))
            .collect();

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
            input_connections,
            output_connections,
            chunks,
        }
    }

    /// Reassemble the chunks into a circuit.
    pub fn reassemble(self) -> Result<Hugr, HugrError> {
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

        // The chunks input and outputs are each identified with a
        // [`ChunkConnection`]. We collect both sides first, and rewire them
        // after the chunks have been inserted.
        let mut sources: HashMap<ChunkConnection, (Node, Port)> = HashMap::new();
        let mut targets: HashMap<ChunkConnection, Vec<(Node, Port)>> = HashMap::new();

        for (&connection, port) in self
            .input_connections
            .iter()
            .zip(reassembled.node_outputs(reassembled_input))
        {
            reassembled.disconnect(reassembled_input, port)?;
            sources.insert(connection, (reassembled_input, port));
        }
        for (&connection, port) in self
            .output_connections
            .iter()
            .zip(reassembled.node_inputs(reassembled_output))
        {
            targets.insert(connection, vec![(reassembled_output, port)]);
        }

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
        for (connection, (source, source_port)) in sources {
            let Some(tgts) = targets.remove(&connection) else {
                continue;
            };
            for (target, target_port) in tgts {
                reassembled.connect(source, source_port, target, target_port)?;
            }
        }

        Ok(reassembled)
    }

    /// Returns a list of references to the split circuits.
    pub fn circuits(&self) -> impl Iterator<Item = &Hugr> {
        self.chunks.iter().map(|chunk| &chunk.circ)
    }
}

#[cfg(feature = "pyo3")]
#[pymethods]
impl CircuitChunks {
    /// Reassemble the chunks into a circuit.
    #[pyo3(name = "reassemble")]
    fn py_reassemble(&self) -> PyResult<Py<PyAny>> {
        let hugr = self.clone().reassemble()?;
        SerialCircuit::encode(&hugr)?.to_tket1()
    }

    /// Returns clones of the split circuits.
    #[pyo3(name = "circuits")]
    fn py_circuits(&self) -> PyResult<Vec<Py<PyAny>>> {
        self.circuits()
            .map(|hugr| SerialCircuit::encode(hugr)?.to_tket1())
            .collect()
    }

    /// Returns clones of the split circuits.
    #[pyo3(name = "update_circuit")]
    fn py_update_circuit(&mut self, index: usize, new_circ: Py<PyAny>) -> PyResult<()> {
        let hugr = SerialCircuit::_from_tket1(new_circ).decode()?;
        if hugr.circuit_signature() != self.chunks[index].circ.circuit_signature() {
            return Err(PyAttributeError::new_err(
                "The new circuit has a different signature.",
            ));
        }
        self.chunks[index].circ = hugr;
        Ok(())
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

        let mut chunks = CircuitChunks::split(&circ, 3);

        // Rearrange the chunks so nodes are inserted in a new order.
        chunks.chunks.reverse();

        let mut reassembled = chunks.reassemble().unwrap();

        reassembled.infer_and_validate(&REGISTRY).unwrap();
        assert_eq!(circ.circuit_hash(), reassembled.circuit_hash());
    }
}
