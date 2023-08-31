use std::{
    collections::{HashMap, HashSet, VecDeque},
    rc::Rc,
};

use hugr::{
    hugr::{
        rewrite::insert_identity::IdentityInsertion,
        views::{HierarchyView, SiblingGraph},
        CircuitUnit, Rewrite, SimpleReplacementError,
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

    fn node(&self) -> Node {
        self.node
    }

    fn qbs(&self) -> &[Qb] {
        &self.qbs
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
        add_to_slice(&mut slices[free_slice], command);
    }

    slices
}

/// Starting from starting_index, work back along slices to check for the
/// earliest slice that can accommodate this command, if any.
fn available_slice(
    circ: &impl HugrView,
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
                blocked_at_slice(command, &slice_vec[slice_index], circ);
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

fn commutation_on_port(comms: &[(usize, Pauli)], port: Port) -> Option<Pauli> {
    comms
        .iter()
        .find_map(|(i, p)| (*i == port.index()).then_some(*p))
}

fn gen_rewrites(
    h: &Hugr,
    next_commands: HashMap<Qb, Rc<Command>>,
    command: &Command,
) -> impl FnOnce(
    &mut Hugr,
) -> Result<
    <SimpleReplacement as Rewrite>::ApplyResult,
    <SimpleReplacement as Rewrite>::Error,
> {
    let remove_node = command.node;

    let replacement = build_simple_circuit(command.qbs.len(), |_circ| Ok(())).unwrap();
    let replace_io = replacement.get_io(replacement.root()).unwrap();
    let nu_inp: HashMap<(Node, Port), (Node, Port)> = h
        .node_inputs(remove_node)
        .zip(replacement.node_inputs(replace_io[1]))
        .map(|(remove_p, replace_p)| ((replace_io[1], replace_p), (remove_node, remove_p)))
        .collect();

    let nu_out: HashMap<(Node, Port), Port> = h
        .node_outputs(remove_node)
        .flat_map(|p| h.linked_ports(remove_node, p))
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

    // insert identities at destination which will get replaced.
    let inserts: Vec<_> = next_commands
        .into_iter()
        .map(|(q, com)| {
            let (node, port) = if &(*com) == command {
                // if there were no nodes between destination and original node,
                // need to connect to original successor.
                let out_port = com.port_of_qb(q, Direction::Outgoing).expect("missing qb");

                h.linked_ports(com.node(), out_port)
                    .exactly_one()
                    .expect("should be linear.")
            } else {
                (
                    com.node,
                    com.port_of_qb(q, Direction::Incoming).expect("missing qb"),
                )
            };

            (q, IdentityInsertion::new(node, port))
        })
        .collect();

    // map from qubits in the original to those in the replacement
    let qb_map: HashMap<Qb, Qb> = command
        .qbs()
        .iter()
        .enumerate()
        .map(|(index, q)| (*q, Qb(index)))
        .collect();

    // replacement circuit containing moved node.
    let replacement = build_simple_circuit(command.qbs().len(), |circ| {
        circ.append(h.get_optype(command.node()).clone(), 0..command.qbs().len())?;
        Ok(())
    })
    .unwrap();

    move |h| {
        // remove node in original location
        h.apply_rewrite(remove)?;

        // add no-ops on all qubit wires in new location
        let noop_nodes: Result<HashMap<Qb, Node>, _> = inserts
            .into_iter()
            .map(|(q, insert)| h.apply_rewrite(insert).map(|res| (q, res)))
            .collect();
        let noop_nodes = noop_nodes.expect("Insert noop failed.");

        let mut nu_inp = HashMap::new();
        let mut nu_out = HashMap::new();

        let replace_io = replacement.get_io(replacement.root()).unwrap();
        // h.validate().unwrap();
        // crate::utils::test::viz_dotstr(&h.dot_string());
        for (q, noop_node) in &noop_nodes {
            let replace_target_port = replacement
                .linked_ports(
                    replace_io[0],
                    PortOffset::new_outgoing(qb_map[q].index()).into(),
                )
                .exactly_one()
                .unwrap();

            let noop_port = h.node_inputs(*noop_node).next().unwrap();

            nu_inp.insert(replace_target_port, (*noop_node, noop_port));

            let noop_target = h
                .linked_ports(*noop_node, PortOffset::new_outgoing(0).into())
                .exactly_one()
                .unwrap();

            let replace_out_port: Port = PortOffset::new_incoming(qb_map[q].index()).into();

            nu_out.insert(noop_target, replace_out_port);
        }

        let replace = SimpleReplacement::new(
            h.root(),
            noop_nodes.into_values().collect(),
            replacement,
            nu_inp,
            nu_out,
        );
        h.apply_rewrite(replace)
    }
}

/// Pass which greedily commutes operations forwards in order to reduce depth.
pub fn apply_greedy_commutation(mut h: Hugr) -> Result<Hugr, SimpleReplacementError> {
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
            if let Some((destination, previous_commands)) =
                available_slice(&h, &slice_vec, slice_index, &command)
            {
                // let subsequent_commands: HashSet<Rc<Command>> =
                //     follow_commands(&slice_vec, destination, &previous_commands);
                for q in &command.qbs {
                    let com = slice_vec[slice_index][q.index()].take();
                    slice_vec[destination][q.index()] = com;
                }
                let rewrite = gen_rewrites(&h, previous_commands, &command);
                rewrite(&mut h)?;
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

    #[fixture]
    // example circuit from original task with lower depth
    fn big_example() -> Hugr {
        build_simple_circuit(4, |circ| {
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
        .unwrap()
    }

    #[fixture]
    // example circuit from original task with lower depth
    fn single_qb_commute() -> Hugr {
        build_simple_circuit(3, |circ| {
            circ.append(T2Op::H, [1])?;
            circ.append(T2Op::CX, [0, 1])?;
            circ.append(T2Op::Z, [0])?;
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
            .iter()
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
        let slices = load_slices(circ);
        let correct = slice_from_command(&commands, 4, &[&[0], &[1], &[2]]);

        assert_eq!(slices, correct);
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

    #[rstest]
    fn big_test(big_example: Hugr) {
        let h = big_example;
        let circ: &SiblingGraph<'_, DfgID> = &SiblingGraph::new(&h, h.root());
        let slices = load_slices(circ);
        assert_eq!(slices.len(), 6);
        // can commute final cx to front
        let (found, prev_nodes) =
            available_slice(&h, &slices, 3, slices[4][1].as_ref().unwrap()).unwrap();
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
        assert!(available_slice(&h, &slices, 4, slices[5][1].as_ref().unwrap()).is_none());
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
            .flat_map(|p| example_cx.linked_ports(remove_node, p))
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

    /// Calculate depth by placing commands in slices.
    fn depth(h: &Hugr) -> usize {
        let circ: &SiblingGraph<'_, DfgID> = &SiblingGraph::new(h, h.root());
        load_slices(circ).len()
    }
    #[rstest]
    #[case(example_cx(), true)]
    #[case(example_cx_better(), false)]
    #[case(big_example(), true)]
    #[case(cant_commute(), false)]
    #[case(t2_bell_circuit(), false)]
    #[case(single_qb_commute(), true)]
    fn commutation_example(#[case] case: Hugr, #[case] should_reduce: bool) {
        let node_count = case.node_count();
        let depth_before = depth(&case);
        let out = apply_greedy_commutation(case).unwrap();

        out.validate().unwrap();

        let depth_after = depth(&out);
        assert!(
            depth_after <= depth_before,
            "Greedy depth optimisation shouldn't ever increase depth."
        );

        if !should_reduce {
            assert_eq!(
                depth_after, depth_after,
                "Depth should not have changed for this case."
            );
        }

        assert_eq!(
            out.node_count(),
            node_count,
            "depth optimisation should not change the number of nodes."
        )
    }
}
