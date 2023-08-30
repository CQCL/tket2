#![allow(unused)]
use std::{
    collections::{HashMap, HashSet, VecDeque},
    rc::Rc,
};

use hugr::{
    hugr::{
        views::{HierarchyView, SiblingGraph},
        CircuitUnit,
    },
    ops::handle::DfgID,
    Direction, Hugr, HugrView, Node, Port, SimpleReplacement,
};
use itertools::Itertools;
use portgraph::PortOffset;

use crate::{
    circuit::{command::Command as CircCommand, Circuit},
    ops::{Pauli, T2Op},
    utils::build_simple_circuit,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct Qb(usize);

impl Qb {
    fn index(&self) -> usize {
        self.0
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct Command {
    node: Node,
    qbs: Vec<Qb>,
}

impl Command {
    fn port_of_qb(&self, qb: Qb, direction: Direction) -> Option<Port> {
        self.qbs
            .iter()
            .position(|q| *q == qb)
            .map(|i| PortOffset::new(direction, i).into())
    }
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
                Qb(i)
            })
            .collect();
        Self { node, qbs }
    }
}

type Slice = Vec<Option<Rc<Command>>>;
type SliceVec = Vec<Slice>;

fn add_to_slice(slice: &mut Slice, com: Rc<Command>) {
    for q in &com.qbs {
        slice[q.index()] = Some(com.clone());
    }
}

fn load_slices<'c>(circ: &'c impl Circuit<'c>) -> SliceVec {
    let mut slices = vec![];
    let mut all_commands: VecDeque<Command> = circ.commands().map_into().collect();

    let n_qbs = circ.units().len();
    let mut qubit_free_slice = vec![0; n_qbs];

    while let Some(command) = all_commands.front() {
        let free_slice = command
            .qbs
            .iter()
            .map(|qb| qubit_free_slice[qb.index()])
            .max()
            .unwrap();

        for q in &command.qbs {
            qubit_free_slice[q.index()] = free_slice + 1;
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
    circ: &'c impl HugrView,
    slice_vec: &[Slice],
    starting_index: usize,
    command: &Rc<Command>,
) -> Option<(usize, HashMap<Qb, Rc<Command>>)> {
    let mut available = None;
    let mut prev_nodes: HashMap<Qb, Rc<Command>> = HashMap::new();

    for slice_index in (0..starting_index + 1).rev() {
        // if all qubit slots are empty here the command can be moved here
        if command
            .qbs
            .iter()
            .all(|q| slice_vec[slice_index][q.index()].is_none())
        {
            available = Some((slice_index, prev_nodes.clone()));
        } else if slice_index == 0 {
            break;
        } else {
            // if command commutes with all ports here it can be moved past,
            // otherwise stop
            let (blocked, new_prev_nodes) =
                blocked_at_slice(&command, &slice_vec[slice_index], circ);
            if blocked {
                break;
            } else {
                prev_nodes.extend(new_prev_nodes);
            }
        }
    }

    available
}

// check if command wouldn't commute through this slice.
fn blocked_at_slice(
    command: &Rc<Command>,
    slice: &Slice,
    circ: &impl HugrView,
) -> (bool, HashMap<Qb, Rc<Command>>) {
    // map from qubit to node it is connected to immediately after the free slice.
    let mut prev_nodes: HashMap<Qb, Rc<Command>> =
        HashMap::from_iter(command.qbs.iter().map(|q| (*q, command.clone())));
    let blocked = command.qbs.iter().enumerate().any(|(port, q)| {
        if let Some(other_com) = &slice[q.index()] {
            let Ok(other_op): Result<T2Op, _> = circ.get_optype(other_com.node).clone().try_into()
            else {
                return true;
            };

            let Ok(op): Result<T2Op, _> = circ.get_optype(command.node).clone().try_into() else {
                return true;
            };

            let port = PortOffset::new_incoming(port).into();
            let Some(pauli) = commutation_on_port(&op.qubit_commutation(), port) else {
                return true;
            };
            let Some(other_pauli) = commutation_on_port(
                &other_op.qubit_commutation(),
                other_com.port_of_qb(*q, Direction::Outgoing).unwrap(),
            ) else {
                return true;
            };

            if pauli.commutes_with(other_pauli) {
                prev_nodes.insert(*q, other_com.clone());
                false
            } else {
                true
            }
        } else {
            false
        }
    });
    (blocked, prev_nodes)
}

fn commutation_on_port(comms: &Vec<(usize, Pauli)>, port: Port) -> Option<Pauli> {
    comms
        .iter()
        .find_map(|(i, p)| (*i == port.index()).then_some(*p))
}

fn gen_rewrites(
    h: &Hugr,
    previous_nodes: &HashMap<Qb, Rc<Command>>,
    command: &Command,
) -> [SimpleReplacement; 2] {
    let remove_node = command.node;

    let op = h.get_optype(remove_node).clone();

    let replacement = build_simple_circuit(2, |_circ| Ok(())).unwrap();
    let replace_io = replacement.get_io(replacement.root()).unwrap();
    let nu_inp: HashMap<(Node, Port), (Node, Port)> = h
        .node_inputs(remove_node)
        .zip(replacement.node_inputs(replace_io[1]))
        .map(|(remove_p, replace_p)| ((replace_io[1], replace_p), (remove_node, remove_p)))
        .collect();

    let nu_out: HashMap<(Node, Port), Port> = h
        .node_outputs(remove_node)
        .map(|p| h.linked_ports(remove_node, p))
        .flatten()
        .zip(replacement.node_inputs(replace_io[1]))
        .map(|(remove_np, replace_p)| (remove_np, replace_p))
        .collect();

    let remove = SimpleReplacement::new(
        h.root(),
        HashSet::from_iter([remove_node]),
        replacement,
        nu_inp,
        nu_out,
    );

    // let next_nodes: HashMap<Qb, Command> = previous_nodes.iter().map(|(qb, node)| {});

    todo!()
}

fn solve(mut h: Hugr) -> Result<Hugr, ()> {
    let circ: &SiblingGraph<'_, DfgID> = &SiblingGraph::new(&h, h.root());
    let mut slice_vec = load_slices(circ);

    for slice_index in 0..slice_vec.len() {
        let slice_commands: Vec<_> = slice_vec[slice_index]
            .iter()
            .flatten()
            .unique()
            .cloned()
            .collect();

        for command in slice_commands {
            if let Some((destination, previous_nodes)) =
                available_slice(&h, &slice_vec, slice_index, &command)
            {
                for q in &command.qbs {
                    let com = slice_vec[slice_index][q.index()].take();
                    slice_vec[destination][q.index()] = com;
                }

                let rewrites: [SimpleReplacement; 2] = gen_rewrites(&h, &previous_nodes, &command);
                for rw in rewrites {
                    h.apply_rewrite(rw).unwrap();
                }
            }
        }
    }

    // TODO remove empty slices and return
    // and full slices at start?
    Ok(h)
}

#[cfg(test)]
mod test {
    use std::collections::HashMap;

    use crate::ops::test::t2_bell_circuit;
    use hugr::{Hugr, Port};
    use itertools::Itertools;
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

    #[fixture]
    // can't commute anything here
    fn cant_commute() -> Hugr {
        build_simple_circuit(3, |circ| {
            circ.append(T2Op::Z, [1])?;
            circ.append(T2Op::CX, [0, 1])?;
            circ.append(T2Op::CX, [2, 1])?;
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
        let commands: Vec<Command> = circ.commands().map_into().collect();

        let slices = load_slices(circ);
        let correct = slice_from_command(&commands, 4, &[&[0, 1], &[2]]);

        assert_eq!(slices, correct);
    }

    #[rstest]
    fn test_load_slices_bell(t2_bell_circuit: Hugr) {
        let circ: &SiblingGraph<'_, DfgID> =
            &SiblingGraph::new(&t2_bell_circuit, t2_bell_circuit.root());
        let commands: Vec<Command> = circ.commands().map_into().collect();

        let slices = load_slices(circ);
        let correct = slice_from_command(&commands, 2, &[&[0], &[1]]);

        assert_eq!(slices, correct);
    }

    #[rstest]
    fn test_available_slice(example_cx: Hugr) {
        let circ: &SiblingGraph<'_, DfgID> = &SiblingGraph::new(&example_cx, example_cx.root());
        let slices = load_slices(circ);
        let (found, prev_nodes) =
            available_slice(&example_cx, &slices, 1, slices[2][1].as_ref().unwrap()).unwrap();
        assert_eq!(found, 0);

        assert_eq!(
            *prev_nodes.get(&Qb(1)).unwrap(),
            slices[1][1].as_ref().unwrap().clone()
        );

        assert_eq!(
            *prev_nodes.get(&Qb(3)).unwrap(),
            slices[2][3].as_ref().unwrap().clone()
        );
    }

    #[test]
    fn test_available_slice_bigger() {
        let example_cx = build_simple_circuit(4, |circ| {
            circ.append(T2Op::CX, [0, 3])?;
            circ.append(T2Op::CX, [1, 2])?;
            circ.append(T2Op::H, [0])?;
            circ.append(T2Op::H, [3])?;
            circ.append(T2Op::CX, [0, 1])?;
            circ.append(T2Op::CX, [2, 3])?;
            circ.append(T2Op::CX, [0, 1])?;
            circ.append(T2Op::CX, [2, 3])?;
            circ.append(T2Op::CX, [2, 1])?;
            circ.append(T2Op::H, [1])?;
            Ok(())
        })
        .unwrap();

        let circ: &SiblingGraph<'_, DfgID> = &SiblingGraph::new(&example_cx, example_cx.root());
        let slices = load_slices(circ);

        // can commute final cx to front
        let (found, prev_nodes) =
            available_slice(&example_cx, &slices, 3, slices[4][1].as_ref().unwrap()).unwrap();
        assert_eq!(found, 1);
        assert_eq!(
            *prev_nodes.get(&Qb(1)).unwrap(),
            slices[2][1].as_ref().unwrap().clone()
        );

        assert_eq!(
            *prev_nodes.get(&Qb(2)).unwrap(),
            slices[2][2].as_ref().unwrap().clone()
        );
        // hadamard can't commute past anything
        assert!(available_slice(&example_cx, &slices, 4, slices[5][1].as_ref().unwrap()).is_none());
        solve(example_cx).unwrap();
    }

    #[rstest]
    fn commutation_simple_bell(t2_bell_circuit: Hugr) {
        solve(t2_bell_circuit).unwrap();
    }

    #[rstest]
    fn test_example_node_removal(mut example_cx: Hugr) {
        assert_eq!(example_cx.node_count(), 6);

        let nodes: Vec<_> = example_cx.nodes().collect();
        let remove_node = nodes[5];

        let replacement = build_simple_circuit(2, |_circ| Ok(())).unwrap();
        let replace_io = replacement.get_io(replacement.root()).unwrap();
        let nu_inp: HashMap<(Node, Port), (Node, Port)> = example_cx
            .node_inputs(remove_node)
            .zip(replacement.node_inputs(replace_io[1]))
            .map(|(remove_p, replace_p)| ((replace_io[1], replace_p), (remove_node, remove_p)))
            .collect();

        let nu_out: HashMap<(Node, Port), Port> = example_cx
            .node_outputs(remove_node)
            .map(|p| example_cx.linked_ports(remove_node, p))
            .flatten()
            .zip(replacement.node_inputs(replace_io[1]))
            .map(|(remove_np, replace_p)| (remove_np, replace_p))
            .collect();

        let rw = SimpleReplacement::new(
            example_cx.root(),
            HashSet::from_iter([nodes[5]]),
            replacement,
            nu_inp,
            nu_out,
        );

        example_cx.apply_rewrite(rw).unwrap();
        assert_eq!(example_cx.node_count(), 5);
    }

    #[rstest]
    fn commutation_example(cant_commute: Hugr) {
        solve(cant_commute).unwrap();
    }
}
