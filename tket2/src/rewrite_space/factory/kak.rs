//! KAKResynthesis explorer for rewrite space exploration.
use crate::{
    rewrite_space::{CommitFactory, IterMatchedWires, PersistentHugr, Walker},
    Tk2Op,
    ops::op_matches,
};
use hugr::persistent::{PatchNode, PersistentWire};
use hugr::Direction;
use hugr::{
    builder::{endo_sig, DFGBuilder, Dataflow, DataflowHugr},
    extension::{prelude::qb_t, simple_op::MakeExtensionOp},
    ops::OpType,
    Hugr, HugrView, IncomingPort, OutgoingPort,
};
use rstest::{fixture, rstest};

#[derive(Clone)]
pub struct KAKResynthesis {
    wires: Vec<PersistentWire>,
}

impl IterMatchedWires for KAKResynthesis {
    fn matched_wires(&self) -> impl Iterator<Item = &PersistentWire> + '_ {
        self.wires.iter()
    }
}

pub struct KAKFactory;

#[derive(Clone)]
struct SearchingWalker<'w> {
    // Initial state of walker.
    walker: Walker<'w>,                  
    // Wires to be expanded.
    frontier_wires: Vec<PersistentWire>, 
    // Current pattern.
    pattern: KAKResynthesis,             
    // If a 2q gate has been found along one wire,
    // but not yet the other, then it is stored here.
    two_qubit_node: Option<PatchNode>,   
}

impl CommitFactory for KAKFactory {
    type PatternMatch = KAKResynthesis;

    type Cost = usize;

    const PATTERN_RADIUS: usize = 5;

    fn find_pattern_matches<'w: 'a, 'a>(
        &'a self,
        pattern_root: PatchNode,
        walker: Walker<'w>,
    ) -> impl Iterator<Item = (Self::PatternMatch, Walker<'w>)> + 'a {
        // A list of matches to be returned.
        let mut matches: Vec<(Self::PatternMatch, Walker<'w>)> = Vec::new();

        // If the node is not a CZ gate, the matches are empty.
        if walker.as_hugr_view().get_optype(pattern_root) != &Tk2Op::CZ.into() {
            return matches.into_iter();
        }

        // A list of all the searching walkers.
        let mut searching_walker_list = Vec::new();
        // Create an initial searching walker, starting at the CZ gate discovered.
        searching_walker_list.push(SearchingWalker {
            walker: walker.clone(),
            frontier_wires: vec![
                walker.get_wire(pattern_root, OutgoingPort::from(0)),
                walker.get_wire(pattern_root, OutgoingPort::from(1)),
            ],
            pattern: KAKResynthesis { wires: Vec::new() },
            two_qubit_node: None,
        });

        // Continue for as long as there are searching walkers to explore.
        while let Some(mut searching_walker) = searching_walker_list.pop() {
            // If there are no wires to be searched then the match gets pushed
            // to the list of matches.
            let Some(wire_to_search) = searching_walker.frontier_wires.pop() else {
                matches.push((searching_walker.pattern, searching_walker.walker));
                continue;
            };

            // Assuming now that there are wires to be searched, get
            // the input port of the wire. There should be exactly one.
            let (frontier_node, frontier_port) = searching_walker
                .walker
                .wire_pinned_outport(&wire_to_search)
                .expect("error");

            // Expand the wire into the future
            for expanded_walker in searching_walker
                .walker//.clone()
                .expand(&wire_to_search, Direction::Incoming)
            {
                // Get the new expanded version of the wire to search along.
                let expanded_wire = expanded_walker.get_wire(frontier_node, frontier_port);

                // Get the node at the end of this newly expanded wire.
                let (expanded_node, _) = expanded_walker
                    .wire_pinned_outport(&expanded_wire)
                    .expect("error");

                // Get the type of operation this node implements
                let ext_op = expanded_walker
                    .as_hugr_view()
                    .get_optype(expanded_node)
                    .as_extension_op()
                    .expect("Not extension op");
                let Ok(tket2_op) = Tk2Op::from_extension_op(ext_op) else {
                    searching_walker_list.push(searching_walker.clone());
                    continue
                };

                // If it it not a quantum operation then there is no need to continue
                // search as this is not a 2q sub circuit.
                if !tket2_op.is_quantum() {
                    searching_walker_list.push(searching_walker.clone());
                // If it is a one qubit operation then we should continue searching.
                // We add the searching walker back to the queue, updated as required.
                } else if n_qubits(expanded_node, expanded_walker.as_hugr_view()) == 1 {
                    let mut new_searching_walker = searching_walker.clone();
                    new_searching_walker
                        .pattern
                        .wires
                        .push(expanded_wire);
                    new_searching_walker
                        .frontier_wires
                        .push(expanded_walker.get_wire(expanded_node, OutgoingPort::from(0)));
                    new_searching_walker.walker = expanded_walker;

                    searching_walker_list.push(new_searching_walker);

                // If it is a 2q operation then the action depends on which
                // qubits the gate acts on.
                // } else if walker.as_hugr_view().get_optype(expanded_node) == Tk2Op::CZ.into() {
                } else if op_matches(walker.as_hugr_view().get_optype(expanded_node), Tk2Op::CZ){
                    // If a 2q gate has already been found by this searcher,
                    // then we need to check if we have found the same one again.
                    if let Some(two_qubit_node) = searching_walker.two_qubit_node {
                        // If the one just found is the same 2q gate that this searcher has already
                        // found, then we have reached the end of a 2q subcircuit.
                        // We add the pattern to our matches, and add the
                        // search back to the queue to continue exploring
                        // along the outputs of the new 2q gate.
                        // If it is not the same then we don't need to do
                        // anything as this is not a 2q sub circuit.
                        if two_qubit_node == expanded_node {
                            let mut new_searching_walker = searching_walker.clone();
                            new_searching_walker.pattern.wires.push(expanded_wire);

                            matches.push((new_searching_walker.pattern.clone(), new_searching_walker.walker.clone()));

                            new_searching_walker.two_qubit_node = None;
                            new_searching_walker.frontier_wires = vec![
                                expanded_walker.get_wire(two_qubit_node, OutgoingPort::from(0)),
                                expanded_walker.get_wire(two_qubit_node, OutgoingPort::from(1)),
                            ];
                            new_searching_walker.walker = expanded_walker;

                            searching_walker_list.push(new_searching_walker);
                        }

                    // If we have not found this 2q gate before then we need
                    // to explore the other wire to see if it meets at
                    // the same place. As such we put the searcher
                    // back on the queue.
                    } else {
                        searching_walker.two_qubit_node = Some(expanded_node);
                        searching_walker_list.push(searching_walker.clone());
                    }
                }
            }
        }

        matches.into_iter()
    }

    fn op_cost(&self, op: &hugr::ops::OpType) -> Option<Self::Cost> {
        todo!()
    }

    fn get_replacement(
        &self,
        pattern_match: &Self::PatternMatch,
        host: &PersistentHugr,
    ) -> Option<Hugr> {
        todo!()
    }

    fn map_boundary(
        &self,
        node: PatchNode,
        port: hugr::Port,
        pattern_match: &Self::PatternMatch,
        host: &PersistentHugr,
    ) -> hugr::Port {
        todo!()
    }

    fn get_name(&self, pattern_match: &Self::PatternMatch, host: &PersistentHugr) -> String {
        todo!()
    }
}

/// only call this on nodes that are known quantum gates
fn n_qubits<N: Copy>(node: N, hugr: &impl HugrView<Node = N>) -> usize {
    debug_assert_eq!(
        hugr.signature(node).unwrap().input_count(),
        hugr.signature(node).unwrap().output_count(),
        "expected a quantum gate"
    );
    hugr.signature(node).unwrap().input_count()
}

#[fixture]
fn empty_2qb_hugr() -> Hugr {
    let builder = DFGBuilder::new(endo_sig(vec![qb_t(), qb_t()])).unwrap();
    let [q0, q1] = builder.input_wires_arr();
    builder.finish_hugr_with_outputs(vec![q0, q1]).unwrap()
}

#[fixture]
fn one_cz_2qb_hugr() -> Hugr {
    let mut builder = DFGBuilder::new(endo_sig(vec![qb_t(), qb_t()])).unwrap();
    let [q0, q1] = builder.input_wires_arr();
    let cz1 = builder.add_dataflow_op(Tk2Op::CZ, vec![q0, q1]).unwrap();
    let [q0, q1] = cz1.outputs_arr();
    builder.finish_hugr_with_outputs(vec![q0, q1]).unwrap()
}

fn three_cz_hugr() -> Hugr {
    let mut builder = DFGBuilder::new(endo_sig(vec![qb_t(); 3])).unwrap();
    let [q0, q1, q2] = builder.input_wires_arr();
    let cz1 = builder.add_dataflow_op(Tk2Op::CZ, vec![q0, q1]).unwrap();
    let [q0, q1] = cz1.outputs_arr();
    let cz2 = builder.add_dataflow_op(Tk2Op::CZ, vec![q0, q1]).unwrap();
    let [q0, q1] = cz2.outputs_arr();
    let cz3 = builder.add_dataflow_op(Tk2Op::CZ, vec![q0, q2]).unwrap();
    let [q0, q2] = cz3.outputs_arr();
    builder.finish_hugr_with_outputs(vec![q0, q1, q2]).unwrap()
}

#[rstest]
fn test_kak_resynthesis_init(one_cz_2qb_hugr: Hugr) {
    let pers_hugr = PersistentHugr::with_base(one_cz_2qb_hugr);

    for node in pers_hugr.nodes() {
        let walker = Walker::from_pinned_node(node, pers_hugr.as_state_space());

        if pers_hugr.get_optype(node) == &Tk2Op::CZ.into() {
            println!("{}", node);
            KAKResynthesis {
                wires: vec![
                    walker.get_wire(node, OutgoingPort::from(0)),
                    walker.get_wire(node, OutgoingPort::from(1)),
                ],
            };
        }
    }
}

#[rstest]
fn test_find_pattern_matches(one_cz_2qb_hugr: Hugr) {
    let hugr = one_cz_2qb_hugr;

    println!("{}", hugr.mermaid_string());

    let factory = KAKFactory;

    let pers_hugr = PersistentHugr::with_base(hugr);
    for node in pers_hugr.nodes() {
        let walker = Walker::from_pinned_node(node, pers_hugr.as_state_space());
        // walker.get_wire(node, OutgoingPort::from(0));
        factory.find_pattern_matches(node, walker);
    }
}
