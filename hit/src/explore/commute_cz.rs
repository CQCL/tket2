//! CommuteCZ explorer for rewrite space exploration.

use hugr::{
    builder::{endo_sig, DFGBuilder, Dataflow, DataflowHugr},
    extension::prelude::qb_t,
    ops::OpType,
    Direction, Hugr, HugrView, IncomingPort, OutgoingPort, Port, PortIndex,
};
use hugr_core::hugr::persistent::{PatchNode, PinnedWire, Walker};
use itertools::{Either, Itertools};
use tket2::{
    op_matches,
    rewrite_space::{Explore, IterMatchedWires},
    Tk2Op,
};

pub struct CommuteCZExplorer;

pub enum CommuteCZ {
    /// Cancel the two CZ gates at both ends of the two parallel wires.
    Cancel([PinnedWire; 2]),
    /// Swap the two CZ gates at both ends of the wire.
    Swap(PinnedWire),
}

impl CommuteCZ {
    fn try_new(wire: PinnedWire, walker: &Walker) -> Option<Self> {
        if !wire.is_complete(None) {
            // only create matches on complete wires
            return None;
        }

        let hugr = walker.as_hugr_view();
        let (out_node, _) = wire
            .pinned_outport()
            .expect("outgoing port was already pinned (and is unique)");

        let (in_node, in_port) = wire
            .pinned_inports()
            .exactly_one()
            .ok()
            .expect("all our wires have exactly one incoming port");

        if hugr.get_optype(out_node) != &Tk2Op::CZ.into()
            || hugr.get_optype(in_node) != &Tk2Op::CZ.into()
        {
            // not 2x CZ gates => no match
            return None;
        }

        // figure out whether the two CZ gates act on the same qubits (iff the
        // the only outgoing neighbour of the first CZ is the second CZ gate)
        let all_edges = hugr.node_connections(out_node, in_node).collect_vec();
        let n_shared_qubits = all_edges.len();

        match n_shared_qubits {
            1 => {
                // the two CZ gates share exactly one qubit => swap them
                Some(CommuteCZ::Swap(wire))
            }
            2 => {
                // the two CZ gates share both qubits => cancel them
                let (wire_out_node, wire_out_port) = wire.pinned_outport().expect("is complete");
                let other_wire =
                    walker.get_wire(wire_out_node, OutgoingPort::from(1 - wire_out_port.index()));
                // we expect the other wire to connect the same nodes, on the other port
                debug_assert!(other_wire.is_complete(None));
                debug_assert_eq!(
                    other_wire.pinned_inports().exactly_one().ok(),
                    Some((in_node, (1 - in_port.index()).into()))
                );
                Some(CommuteCZ::Cancel([wire, other_wire]))
            }
            _ => {
                // the two CZ gates share more than two qubits => no match
                panic!("invalid hugr");
            }
        }
    }

    fn wires(&self) -> impl Iterator<Item = &PinnedWire> + '_ {
        match self {
            CommuteCZ::Cancel(wires) => Either::Left(wires.iter()),
            CommuteCZ::Swap(wire) => Either::Right(std::iter::once(wire)),
        }
    }

    /// The first of the two CZ gates
    fn out_node(&self) -> PatchNode {
        self.wires()
            .map(|w| w.pinned_outport().expect("is complete"))
            .unique()
            .exactly_one()
            .ok()
            .expect("invalid CommuteCZ match")
            .0
    }
}

impl IterMatchedWires for CommuteCZ {
    fn matched_wires(&self) -> impl Iterator<Item = &PinnedWire> + '_ {
        self.wires().into_iter()
    }
}

impl Explore for CommuteCZExplorer {
    type PatternMatch = CommuteCZ;

    type Cost = usize;

    const PATTERN_RADIUS: usize = 1;

    fn get_replacement(&self, pattern_match: &Self::PatternMatch) -> Option<Hugr> {
        match pattern_match {
            CommuteCZ::Swap(_) => Some(two_cz_3qb_hugr()),
            CommuteCZ::Cancel(wires) => {
                let add_swap = out_port(&wires[0]).index() != in_port(&wires[0]).index();
                Some(empty_2qb_hugr(add_swap))
            }
        }
    }

    fn map_boundary(
        &self,
        node: PatchNode,
        port: Port,
        pattern_match: &Self::PatternMatch,
    ) -> hugr::Port {
        match pattern_match {
            CommuteCZ::Cancel(_) => {
                // the incoming/outgoing ports of the subgraph map trivially to the empty 2qb
                // HUGR
                let dir = port.direction();
                Port::new(dir.reverse(), port.index())
            }
            CommuteCZ::Swap(wire) => {
                // map the incoming/outgoing ports of the subgraph to the replacement as
                // follows:
                //  - the first qubit is the one that is shared between the two CZ gates
                //  - the second qubit only touches the first CZ (out_node)
                //  - the third qubit only touches the second CZ (in_node)
                let shared_qb_out = out_port(wire).index();
                let shared_qb_in = in_port(wire).index();
                match (port.as_directed(), node == pattern_match.out_node()) {
                    (Either::Left(incoming), true) if incoming.index() == shared_qb_out => {
                        // out_node on the shared qubit -> port 0
                        OutgoingPort::from(0).into()
                    }
                    (Either::Left(incoming), true) if incoming.index() == 1 - shared_qb_out => {
                        // out_node on the not shared qubit -> port 1
                        OutgoingPort::from(1).into()
                    }
                    (Either::Left(incoming), false) if incoming.index() == 1 - shared_qb_in => {
                        // in_node on the not shared qubit -> port 2
                        OutgoingPort::from(2).into()
                    }
                    (Either::Right(outgoing), false) if outgoing.index() == shared_qb_in => {
                        // in_node on the shared qubit -> port 0
                        IncomingPort::from(0).into()
                    }
                    (Either::Right(outgoing), true) if outgoing.index() == 1 - shared_qb_out => {
                        // out_node on the not shared qubit -> port 1
                        IncomingPort::from(1).into()
                    }
                    (Either::Right(outgoing), false) if outgoing.index() == 1 - shared_qb_in => {
                        // in_node on the not shared qubit -> port 2
                        IncomingPort::from(2).into()
                    }
                    _ => panic!("unexpected boundary port"),
                }
            }
        }
    }

    fn find_pattern_matches<'w: 'a, 'a>(
        &'a self,
        pattern_root: PatchNode,
        walker: Walker<'w>,
    ) -> impl Iterator<Item = (Self::PatternMatch, Walker<'w>)> + 'a {
        let mut matches: Vec<(Self::PatternMatch, Walker<'w>)> = Vec::new();
        if walker.as_hugr_view().get_optype(pattern_root) != &Tk2Op::CZ.into() {
            return matches.into_iter();
        }

        for out_port in (0..2).map(OutgoingPort::from) {
            let wire = walker.get_wire(pattern_root, out_port);
            let complete_wires = walker
                .expand(&wire, Direction::Incoming)
                .map(move |subwalker| {
                    let new_wire = subwalker.get_wire(pattern_root, out_port);
                    (subwalker, new_wire)
                });
            let complete_wires = complete_wires.collect_vec();
            for (subwalker, wire) in complete_wires {
                let in_node = wire.pinned_inports().exactly_one().ok().unwrap().0;
                if subwalker.as_hugr_view().get_optype(in_node) != &Tk2Op::CZ.into() {
                    continue;
                }
                let match_ = CommuteCZ::try_new(wire, &subwalker).expect("failed creating match");
                matches.push((match_, subwalker));
            }
        }
        matches.into_iter()
    }

    fn op_cost(&self, op: &OpType) -> Option<Self::Cost> {
        op_matches(op, Tk2Op::CZ).then_some(1)
    }
}

fn empty_2qb_hugr(flip_args: bool) -> Hugr {
    let builder = DFGBuilder::new(endo_sig(vec![qb_t(), qb_t()])).unwrap();
    let [mut q0, mut q1] = builder.input_wires_arr();
    if flip_args {
        (q0, q1) = (q1, q0);
    }
    builder.finish_hugr_with_outputs(vec![q0, q1]).unwrap()
}

fn two_cz_3qb_hugr() -> Hugr {
    let mut builder = DFGBuilder::new(endo_sig(vec![qb_t(), qb_t(), qb_t()])).unwrap();
    let [q0, q1, q2] = builder.input_wires_arr();
    let cz1 = builder.add_dataflow_op(Tk2Op::CZ, vec![q0, q2]).unwrap();
    let [q0, q2] = cz1.outputs_arr();
    let cz2 = builder.add_dataflow_op(Tk2Op::CZ, vec![q0, q1]).unwrap();
    let [q0, q1] = cz2.outputs_arr();
    builder.finish_hugr_with_outputs(vec![q0, q1, q2]).unwrap()
}

fn out_port(w: &PinnedWire) -> OutgoingPort {
    w.pinned_outport().unwrap().1.index().into()
}

fn in_port(w: &PinnedWire) -> IncomingPort {
    w.pinned_inports()
        .exactly_one()
        .ok()
        .unwrap()
        .1
        .index()
        .into()
}
