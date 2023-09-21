//! Utility

use hugr::extension::ExtensionSet;
use hugr::hugr::views::sibling_subgraph::ConvexChecker;
use hugr::hugr::views::SiblingSubgraph;
use hugr::{Hugr, HugrView, Node, Wire};
use itertools::Itertools;

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

impl Chunk {
    /// Extract a chunk from a circuit.
    ///
    /// The chunk is extracted from the input wires to the output wires.
    pub fn extract<'h, H: HugrView>(
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
        let circ = subgraph
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
                    circ.linked_ports(inp_node, inp_port).exactly_one().unwrap();
                Wire::new(out_node, out_port)
            })
            .collect();
        let outputs = subgraph
            .outgoing_ports()
            .iter()
            .map(|&(node, port)| Wire::new(node, port))
            .collect();
        Self {
            circ,
            inputs,
            outputs,
        }
    }
}

/// An utility for splitting a circuit into chunks, and reassembling them afterwards.
#[derive(Debug, Clone)]
pub struct CircuitChunks {
    /// The split circuits.
    pub chunks: Vec<Chunk>,
}

impl CircuitChunks {
    /// Split a circuit into chunks.
    ///
    /// The circuit is split into chunks of at most `max_size` gates.
    pub fn split(circ: &impl Circuit, max_size: usize) -> Self {
        let mut chunks = Vec::new();
        let mut convex_checker = ConvexChecker::new(circ);
        for commands in &circ.commands().chunks(max_size) {
            chunks.push(Chunk::extract(
                circ,
                commands.map(|cmd| cmd.node()),
                &mut convex_checker,
            ));
        }
        Self { chunks }
    }

    /// Reassemble the chunks into a circuit.
    pub fn reassemble(self) -> Hugr {
        todo!()
    }
}
