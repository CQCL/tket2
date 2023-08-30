#![allow(unused)]
use std::{
    collections::{HashSet, VecDeque},
    rc::Rc,
};

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

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
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

type Slice = Vec<Option<Rc<Command>>>;
type SliceVec = Vec<Slice>;

fn add_to_slice(slice: &mut Slice, com: Rc<Command>) {
    for q in &com.qbs {
        slice[*q] = Some(com.clone());
    }
}

fn load_slices<'c>(circ: &'c impl Circuit<'c>) -> SliceVec {
    let mut slices = vec![];
    let mut cur_index = 0;
    let mut all_commands: VecDeque<Command> = circ.commands().map_into().collect();

    let n_qbs = circ.units().len();
    let mut qubit_free_slice = vec![0; n_qbs];

    while let Some(command) = all_commands.front() {
        let free_slice = command
            .qbs
            .iter()
            .map(|qb| qubit_free_slice[*qb])
            .max()
            .unwrap();

        for q in &command.qbs {
            qubit_free_slice[*q] = free_slice + 1;
        }
        if free_slice >= slices.len() {
            debug_assert!(free_slice == slices.len());
            slices.push(vec![None; n_qbs]);
        }
        let command = Rc::new(all_commands.pop_front().unwrap());
        let qbs = command.qbs.clone();
        add_to_slice(&mut slices[free_slice], command);
    }

    slices
}

fn gen_rewrite<'c>(circ: &'c impl HugrView, commute_nodes: [Node; 2]) -> SimpleReplacement {
    todo!()
}

/// Starting from given index, search the slices for the command for the given
/// node, returning it and the slice at which it was found
fn find_command(
    slice_vec: &[Slice],
    starting_index: usize,
    node: Node,
) -> Option<(Rc<Command>, usize)> {
    for slice_index in starting_index..slice_vec.len() {
        let slice = slice_vec.get(slice_index).unwrap();
        if let Some(command) = slice.iter().flatten().find(|c| c.node == node) {
            return Some((command.clone(), slice_index));
        }
    }
    None
}

/// Starting from starting_index, work back along slices to check for the
/// earliest slice that can accommodate this command, if any.
fn available_slice<'c>(
    slice_vec: &[Slice],
    starting_index: usize,
    other: Rc<Command>,
) -> Option<usize> {
    let mut slice_index = starting_index;

    let qbs: HashSet<_> = HashSet::from_iter(&other.qbs);
    loop {
        if other
            .qbs
            .iter()
            .any(|q| !slice_vec[slice_index][*q].is_none())
        {
            if slice_index == starting_index {
                return None;
            } else {
                return Some(slice_index + 1);
            }
        }

        if slice_index == 0 {
            return Some(0);
        }
        slice_index -= 1;
    }

    None
}

fn commutation_on_port(comms: &Vec<(usize, Pauli)>, port: usize) -> Option<Pauli> {
    comms.iter().find_map(|(i, p)| (*i == port).then_some(*p))
}

fn solve(mut h: Hugr) -> Result<Hugr, ()> {
    let circ: &SiblingGraph<'_, DfgID> = &SiblingGraph::new(&h, h.root());
    let mut slice_vec = load_slices(circ);

    let mut slice_index: usize = 1;

    loop {
        // keep going until reaching the end of the circuit
        let Some(next_slice) = slice_vec.get(slice_index + 1) else {
            break;
        };
        let search_for_spot =
            find_candidates(&slice_vec[slice_index], &h).find_map(|(command, other_node)| {
                let (other_com, source) = find_command(&slice_vec, slice_index + 1, other_node)?;
                available_slice(&slice_vec, slice_index - 1, other_com.clone())
                    .map(|dest| ([command, other_com], [source, dest]))
            });
        if let Some(([com, other_com], [source, destination])) = search_for_spot {
            let n = com.node;
            let n2 = other_com.node;

            for q in &other_com.qbs {
                slice_vec[source][*q] = None;
                slice_vec[destination][*q] = Some(other_com.clone());
            }
            let rewrite = gen_rewrite(&h, [n, n2]);
            h.apply_rewrite(rewrite).unwrap();
        } else {
            // no candidates left here, move on
            slice_index += 1;
        }
    }
    Ok(h)
}

/// Return pairs of command in current slice and subsequent nodes they commute
/// with (and are connected to).
fn find_candidates<'a, 'c: 'a>(
    current_slice: &'a Slice,
    circ: &'c impl HugrView,
) -> impl Iterator<Item = (Rc<Command>, Node)> + 'a {
    current_slice
        .iter()
        .flatten()
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
                    .then_some((command.clone(), other))
            }))
        })
        .flatten()
        .unique()
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
    // example circuit from original task with lower depth
    fn example_cx_better() -> Hugr {
        build_simple_circuit(4, |circ| {
            circ.append(T2Op::CX, [0, 2])?;
            circ.append(T2Op::CX, [1, 3])?;
            circ.append(T2Op::CX, [1, 2])?;
            Ok(())
        })
        .unwrap()
    }

    fn slice_from_command(
        commands: &Vec<Command>,
        n_qbs: usize,
        slice_arr: &[&[usize]],
    ) -> SliceVec {
        slice_arr
            .into_iter()
            .map(|command_indices| {
                let mut slice = vec![None; n_qbs];
                for ind in command_indices.iter() {
                    let com = commands[*ind].clone();
                    add_to_slice(&mut slice, Rc::new(com))
                }

                slice
            })
            .collect()
    }

    #[rstest]
    fn test_load_slices_cx(example_cx: Hugr) {
        let circ: &SiblingGraph<'_, DfgID> = &SiblingGraph::new(&example_cx, example_cx.root());
        let commands: Vec<Command> = circ.commands().map_into().collect();
        let final_command = commands[2].clone();
        let mut slices = load_slices(circ);
        let correct = slice_from_command(&commands, 4, &[&[0], &[1], &[2]]);

        assert_eq!(slices, correct);

        let found_command = find_command(&mut slices, 0, final_command.node);

        assert_eq!(found_command, Some((Rc::new(final_command), 2)));
    }

    #[rstest]
    fn test_load_slices_cx_better(example_cx_better: Hugr) {
        let circ: &SiblingGraph<'_, DfgID> =
            &SiblingGraph::new(&example_cx_better, example_cx_better.root());
        let mut commands: Vec<Command> = circ.commands().map_into().collect();

        let slices = load_slices(circ);
        let correct = slice_from_command(&commands, 4, &[&[0, 1], &[2]]);

        assert_eq!(slices, correct);
    }

    #[rstest]
    fn test_load_slices_bell(t2_bell_circuit: Hugr) {
        let circ: &SiblingGraph<'_, DfgID> =
            &SiblingGraph::new(&t2_bell_circuit, t2_bell_circuit.root());
        let mut commands: Vec<Command> = circ.commands().map_into().collect();

        let slices = load_slices(circ);
        let correct = slice_from_command(&commands, 2, &[&[0], &[1]]);

        assert_eq!(slices, correct);
    }

    #[rstest]
    fn test_find_candidates(example_cx: Hugr) {
        let circ: &SiblingGraph<'_, DfgID> = &SiblingGraph::new(&example_cx, example_cx.root());
        let nodes: Vec<_> = circ.nodes().collect();
        let slices = load_slices(circ);
        let candidates: Vec<_> = find_candidates(&slices[1], circ).collect();
        dbg!(&candidates);
        let correct: Vec<[Node; 2]> = vec![[nodes[4], nodes[5]]];
        assert!(correct.into_iter().eq(candidates
            .into_iter()
            .map(|(command, node)| [command.node, node])))
    }

    #[rstest]
    fn test_available_slice(example_cx: Hugr) {
        let circ: &SiblingGraph<'_, DfgID> = &SiblingGraph::new(&example_cx, example_cx.root());
        let slices = load_slices(circ);
        let found = available_slice(&slices, 0, slices[2][1].as_ref().cloned().unwrap());
        assert_eq!(found, Some(0));
    }

    #[test]
    fn test_available_slice_bigger() {
        let example_cx = build_simple_circuit(4, |circ| {
            circ.append(T2Op::CX, [0, 3])?;
            circ.append(T2Op::CX, [1, 2])?;
            circ.append(T2Op::H, [0])?;
            circ.append(T2Op::H, [3])?;
            circ.append(T2Op::H, [0])?;
            circ.append(T2Op::H, [3])?;
            circ.append(T2Op::CX, [0, 1])?;
            circ.append(T2Op::CX, [2, 3])?;
            circ.append(T2Op::CX, [2, 1])?;
            circ.append(T2Op::H, [1])?;
            Ok(())
        })
        .unwrap();
        // crate::utils::test::viz_dotstr(&example_cx.dot_string());
        let circ: &SiblingGraph<'_, DfgID> = &SiblingGraph::new(&example_cx, example_cx.root());
        let slices = load_slices(circ);
        let found = available_slice(&slices, 2, slices[4][1].as_ref().cloned().unwrap());
        assert_eq!(found, Some(1));
    }

    #[rstest]
    fn commutation_simple_bell(t2_bell_circuit: Hugr) {
        solve(t2_bell_circuit).unwrap();
    }
}
