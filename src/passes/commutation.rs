#![allow(unused)]
use std::collections::VecDeque;

use hugr::{
    hugr::{
        views::{HierarchyView, SiblingGraph},
        CircuitUnit,
    },
    ops::handle::DfgID,
    Hugr, HugrView, Node, SimpleReplacement,
};
use itertools::Itertools;

use crate::{
    circuit::{command::Command as CircCommand, Circuit},
    ops::{Pauli, T2Op},
};

#[derive(Debug, PartialEq)]
struct Command {
    node: Node,
    qbs: Vec<usize>,
}

impl<'a> From<CircCommand<'a>> for Command {
    fn from(com: CircCommand<'a>) -> Self {
        let CircCommand { node, inputs, .. } = com;
        let qbs = inputs
            .into_iter()
            .map(|u| {
                let CircuitUnit::Linear(i) = u else {
                    panic!("not linear unit.")
                };
                i
            })
            .collect();
        Self { node, qbs }
    }
}

type Slice = Vec<Command>;
type SliceVec = Vec<Slice>;

fn load_slices<'c>(circ: &'c impl Circuit<'c>) -> SliceVec {
    let mut slices = vec![];

    let mut all_commands: VecDeque<Command> = circ.commands().map_into().collect();
    while !all_commands.is_empty() {
        let mut qubit_free = vec![true; circ.units().len()];

        let mut cur_slice = vec![];
        while let Some(command) = all_commands.front() {
            if command.qbs.iter().all(|i| qubit_free[*i]) {
                for q in &command.qbs {
                    qubit_free[*q] = false;
                }
                cur_slice.push(all_commands.pop_front().unwrap());
            } else {
                break;
            }
        }
        slices.push(cur_slice);
    }
    slices
}

fn gen_rewrite<'c>(circ: &'c impl HugrView, commute_nodes: [Node; 2]) -> SimpleReplacement {
    todo!()
}

/// Starting from given index, search the slices for the command for the given
/// node, then remove and return it.
fn remove_command(
    slice_vec: &mut [Vec<Command>],
    starting_index: usize,
    commute_candidate: Node,
) -> Command {
    todo!()
}

/// Starting from starting_index, work back along slices to check for the
/// earliest slice that can accommodate this node.
fn available_slice<'c>(
    circ: &'c impl HugrView,
    slice_vec: &[Vec<Command>],
    starting_index: usize,
    other: Node,
) -> Option<usize> {
    todo!()
}

fn commutation_on_port(comms: &Vec<(usize, Pauli)>, port: usize) -> Option<Pauli> {
    comms.iter().find_map(|(i, p)| (*i == port).then_some(*p))
}

fn solve(mut h: Hugr) -> Result<Hugr, ()> {
    let circ: &SiblingGraph<'_, DfgID> = &SiblingGraph::new(&h, h.root());
    let mut slice_vec = load_slices(circ);

    let mut slice_index: usize = 1;

    let mut done = false;
    loop {
        // keep going until reaching the end of the circuit
        let Some(next_slice) = slice_vec.get(slice_index + 1) else {
            break;
        };
        let search_for_spot = find_candidates(next_slice, &h).find_map(|[n, other]| {
            available_slice(&h, &slice_vec, slice_index - 1, other).map(|dest| ([n, other], dest))
        });
        if let Some((commute_candidate, destination)) = search_for_spot {
            let command: Command =
                remove_command(&mut slice_vec, slice_index + 1, commute_candidate[1]);

            slice_vec[destination].push(command);
            let rewrite = gen_rewrite(&h, commute_candidate);
            h.apply_rewrite(rewrite).unwrap();
        } else {
            // no candidates left here, move on
            slice_index += 1;
        }
    }
    Ok(h)
}

/// Return pairs of nodes, the first in the given slice, which commute.
fn find_candidates<'a, 'c: 'a>(
    current_slice: &'a Vec<Command>,
    circ: &'c impl HugrView,
) -> impl Iterator<Item = [Node; 2]> + 'a {
    current_slice
        .iter()
        .filter_map(move |command| {
            let node = command.node;
            let node_op: T2Op = circ.get_optype(node).clone().try_into().ok()?;
            let node_comm = node_op.qubit_commutation();

            Some(circ.output_neighbours(node).filter_map(move |other| {
                let other_op: T2Op = circ.get_optype(other).clone().try_into().ok()?;
                let other_comm = other_op.qubit_commutation();
                // if the two ops commute on all the ports that they are
                // connected by, then they are a valid commutation pair.
                circ.node_connections(node, other)
                    .all(|[port, other_port]| {
                        commutation_on_port(&node_comm, port.index())
                            .unwrap()
                            .commutes_with(
                                commutation_on_port(&other_comm, other_port.index()).unwrap(),
                            )
                    })
                    .then_some([node, other])
            }))
        })
        .flatten()
}

#[cfg(test)]
mod test {
    use crate::ops::test::{build_simple_circuit, t2_bell_circuit};
    use hugr::Hugr;
    use itertools::Itertools;
    use portgraph::NodeIndex;
    use rstest::{fixture, rstest};

    use super::*;

    #[fixture]
    // example circuit from original task
    fn example_cx() -> Hugr {
        build_simple_circuit(4, |circ| {
            circ.append(T2Op::CX, [0, 2])?;
            circ.append(T2Op::CX, [1, 2])?;
            circ.append(T2Op::CX, [1, 3])?;
            Ok(())
        })
        .unwrap()
    }

    #[fixture]
    // example circuit from original task
    fn example_cx_better() -> Hugr {
        build_simple_circuit(4, |circ| {
            circ.append(T2Op::CX, [0, 2])?;
            circ.append(T2Op::CX, [1, 3])?;
            circ.append(T2Op::CX, [1, 2])?;
            Ok(())
        })
        .unwrap()
    }

    #[rstest]
    fn test_load_slices_cx(example_cx: Hugr) {
        let circ: &SiblingGraph<'_, DfgID> = &SiblingGraph::new(&example_cx, example_cx.root());
        let mut commands: Vec<Command> = circ.commands().map_into().collect();

        let slices = load_slices(circ);
        let correct = vec![
            vec![commands.remove(0)],
            vec![commands.remove(0)],
            vec![commands.remove(0)],
        ];

        assert_eq!(slices, correct);
    }

    #[rstest]
    fn test_load_slices_cx_better(example_cx_better: Hugr) {
        let circ: &SiblingGraph<'_, DfgID> =
            &SiblingGraph::new(&example_cx_better, example_cx_better.root());
        let mut commands: Vec<Command> = circ.commands().map_into().collect();

        let slices = load_slices(circ);
        let correct = vec![
            vec![commands.remove(0), commands.remove(0)],
            vec![commands.remove(0)],
        ];
        assert_eq!(slices, correct);
    }

    #[rstest]
    fn test_load_slices_bell(t2_bell_circuit: Hugr) {
        let circ: &SiblingGraph<'_, DfgID> =
            &SiblingGraph::new(&t2_bell_circuit, t2_bell_circuit.root());
        let mut commands: Vec<Command> = circ.commands().map_into().collect();

        let slices = load_slices(circ);

        let correct = vec![vec![commands.remove(0)], vec![commands.remove(0)]];
        assert_eq!(slices, correct);
    }

    #[rstest]
    fn test_find_candidates(example_cx: Hugr) {
        let circ: &SiblingGraph<'_, DfgID> = &SiblingGraph::new(&example_cx, example_cx.root());
        let nodes: Vec<_> = circ.nodes().collect();
        let slices = load_slices(circ);
        let candidates: Vec<_> = find_candidates(&slices[1], circ).collect();

        let correct: Vec<[Node; 2]> = vec![[nodes[4], nodes[5]]];

        assert_eq!(candidates, correct);
    }

    #[rstest]
    fn commutation_simple_bell(t2_bell_circuit: Hugr) {
        solve(t2_bell_circuit).unwrap();
    }
}
