use crate::circuit::circuit::{Circuit, CircuitRewrite};
use crate::circuit::dag::{Edge, Vertex};
use crate::circuit::operation::{Op, WireType};
use portgraph::substitute::{BoundedSubgraph, SubgraphRef};
use std::collections::HashSet;

use super::{Architecture, Distances, Mapping, Move, QAddr};

fn swap_circ() -> Circuit {
    let mut swapc = Circuit::new();

    let inputs = [(); 2].map(|_| swapc.new_input(WireType::Qubit));
    let outputs = [(); 2].map(|_| swapc.new_output(WireType::Qubit));

    swapc.add_vertex_with_edges(
        Op::Swap,
        [inputs[1], inputs[0]].into(),
        [outputs[0], outputs[1]].into(),
    );

    swapc
}

#[derive(Clone)]
pub(super) struct NodeState {
    pub(super) circ: Circuit,
    pub(super) mapping: Mapping,
}

impl NodeState {
    fn gen_rewrite(&self, mve: Move) -> CircuitRewrite {
        let Move::Swap(swap) = mve;

        let es = swap.map(|qddr| {
            *self
                .mapping
                .get_by_right(&qddr)
                .expect("addr not found in mapping")
        });

        CircuitRewrite::new(
            BoundedSubgraph::new(SubgraphRef::new(HashSet::new()), [es.into(), es.into()]),
            swap_circ().into(),
            0.0,
        )
    }

    pub(super) fn all_moves<'a: 's, 's>(
        &'s self,
        arc: &'a Architecture,
    ) -> impl Iterator<Item = (Move, f64)> + '_ {
        let frontier_physicals: HashSet<_> = self
            .mapping
            .iter()
            .filter_map(|(e, q)| {
                let target = self.circ.edge_endpoints(*e).unwrap().1;
                self.circ
                    .node_op(target)
                    .unwrap()
                    .is_two_qb_gate()
                    .then_some(*q)
                // .map(|qad| (e, qad))
            })
            .collect();
        let mut qddr_pairs = HashSet::new();
        for qddr in frontier_physicals {
            let neighs = arc.neighbors(qddr.0.into());

            for n in neighs {
                // TODO physical qubits not already in mapping are not
                // considered
                let n_addr = QAddr(n.index() as u32);
                if self.mapping.contains_right(&n_addr) {
                    let mut p = [qddr, n_addr];
                    p.sort();
                    qddr_pairs.insert(p);
                }
            }
        }
        qddr_pairs.into_iter().map(|mve| {
            let mve = Move::Swap(mve);
            let mut mapping = move_update(self.mapping.clone(), &mve);

            let reward = advance_frontier(&self.circ, arc, &mut mapping) as f64;
            (mve, reward)
        })
        // // TODO bridges and more
    }

    pub(super) fn child_state(&self, mve: Move, arc: &Architecture) -> Self {
        let rw = self.gen_rewrite(mve.clone());
        let mut circ = self.circ.clone();
        circ.apply_rewrite(rw).expect("rewrite failure");
        let mut mapping = move_update_after_swap(&circ, self.mapping.clone(), &mve);

        advance_frontier(&circ, arc, &mut mapping);
        Self { circ, mapping }
    }

    pub(super) fn sim_heuristic(
        &self,
        mve: &Move,
        dists: &Distances,
        n_layers: usize,
        layer_discount: f64,
    ) -> f64 {
        let mut total = 0.0;
        let mut discount = layer_discount;

        let layiter = TwoqbLayerIter {
            frontier: self.mapping.left_values().cloned().collect(),
            circ: &self.circ,
        };
        let mut mapping = self.mapping.clone();

        for layer in layiter.take(n_layers) {
            total +=
                (layer_dist_change(&self.circ, &layer, &mapping, dists, mve) as f64) * discount;
            discount *= layer_discount;
            mapping = update_layer_mapping(&self.circ, &layer, mapping);
        }
        total
    }

    pub(super) fn solved(&self) -> bool {
        self.mapping.left_values().all(|e| {
            matches!(
                self.circ
                    .node_op(self.circ.edge_endpoints(*e).unwrap().1)
                    .unwrap(),
                Op::Output
            )
        })
    }
}

fn next_edge(circ: &Circuit, n: Vertex, e: Edge) -> (Edge, Vertex) {
    let port = circ
        .port_of_edge(n, e, portgraph::graph::Direction::Incoming)
        .unwrap();
    let e = circ
        .edge_at_port(n, port, portgraph::graph::Direction::Outgoing)
        .expect("Node doesn't have edge at equivalent port.");

    let (_, tgt) = circ.edge_endpoints(e).unwrap();
    (e, tgt)
}

fn move_update(mut mapping: Mapping, mve: &Move) -> Mapping {
    let Move::Swap(qs) = mve;

    let es = qs.map(|q| mapping.remove_by_right(&q).expect("missing in map").0);

    mapping.insert(es[0], qs[1]);
    mapping.insert(es[1], qs[0]);

    mapping
}

fn move_update_after_swap(circ: &Circuit, mut mapping: Mapping, mve: &Move) -> Mapping {
    let Move::Swap(qs) = mve;

    let es = qs.map(|q| mapping.remove_by_right(&q).expect("missing in map").0);

    let (_, swap_node) = circ.edge_endpoints(es[0]).expect("edge not in circuit");
    let outes = es.map(|e| next_edge(circ, swap_node, e).0);

    mapping.insert(outes[1], qs[1]);
    mapping.insert(outes[0], qs[0]);

    mapping
}

fn advance_frontier(circ: &Circuit, arc: &Architecture, mapping: &mut Mapping) -> u64 {
    let mut count = 0;

    loop {
        let mut e_updates: Vec<_> = vec![];
        let mut seen = HashSet::new();
        for (map_in, map_q) in mapping.iter() {
            let (_, tgt) = circ.edge_endpoints(*map_in).expect("edge not in circuit");
            if !seen.insert(tgt) {
                continue;
            }
            let op = circ.node_op(tgt).expect("node missing");
            let out_q_edges: Vec<_> = circ
                .node_edges(tgt, portgraph::graph::Direction::Outgoing)
                .into_iter()
                .filter(|e| circ.edge_type(*e) == Some(WireType::Qubit))
                .collect();
            if op.is_two_qb_gate() {
                let ins = circ.node_edges(tgt, portgraph::graph::Direction::Incoming);
                let e_other = ins.iter().find(|e| *e != map_in).unwrap();
                debug_assert!(ins.contains(map_in));
                let other_q = if let Some(q2) = mapping.get_by_left(e_other) {
                    q2
                } else {
                    // gate not yet at front
                    continue;
                };
                if arc.contains_edge((*map_q).into(), (*other_q).into()) {
                    for p in ins.into_iter().zip(out_q_edges.into_iter()) {
                        e_updates.push(p);
                    }
                    count += 1;
                }
            } else if out_q_edges.len() == 1 {
                e_updates.push((*map_in, out_q_edges.into_iter().next().unwrap()));
            }
        }
        if e_updates.is_empty() {
            return count;
        }
        for (e1, e2) in e_updates {
            let (_, q) = mapping
                .remove_by_left(&e1)
                .expect("edge missing in mapping");
            mapping.insert(e2, q);
        }
    }
}
pub(super) enum NodeStateEnum {
    Root(Box<NodeState>),
    Child(Box<Move>, Option<Box<NodeState>>),
}

impl NodeStateEnum {
    pub(super) fn root_state(circ: Circuit, arc: &Architecture, mut mapping: Mapping) -> Self {
        advance_frontier(&circ, arc, &mut mapping);
        NodeStateEnum::Root(Box::new(NodeState { circ, mapping }))
    }

    pub(super) fn make_root(&mut self) {
        match self {
            NodeStateEnum::Root(_) => (),
            NodeStateEnum::Child(_, x) if x.is_some() => {
                *self = NodeStateEnum::Root(x.take().unwrap())
            }
            _ => panic!("State must be calculated first."),
        };
    }

    pub(super) fn take_nodestate(self) -> Option<NodeState> {
        match self {
            NodeStateEnum::Root(s) => Some(*s),
            NodeStateEnum::Child(_, s) => s.map(|s| *s),
        }
    }
}

struct TwoqbLayerIter<'c> {
    frontier: HashSet<Edge>,
    circ: &'c Circuit,
}

impl<'c> Iterator for TwoqbLayerIter<'c> {
    type Item = HashSet<Vertex>;

    fn next(&mut self) -> Option<Self::Item> {
        let mut out = HashSet::new();

        let mut newfront: HashSet<_> = self
            .frontier
            .iter()
            .map(|e| {
                let mut e = *e;
                loop {
                    let (_, tgt) = self.circ.edge_endpoints(e).expect("edge missing");
                    let op = self.circ.node_op(tgt).expect("nodemissing");

                    // two qb gate with parameter inputs will fail here, compare
                    // edge types directly
                    if op.is_two_qb_gate() {
                        out.insert(tgt);
                        break e;
                    } else if matches!(op, Op::Output) {
                        break e;
                    } else {
                        e = next_edge(self.circ, tgt, e).0;
                    }
                }
            })
            .collect();

        let out: HashSet<_> = out
            .into_iter()
            .filter(|tgt| {
                let ins = self
                    .circ
                    .node_edges(*tgt, portgraph::graph::Direction::Incoming);
                ins.iter().all(|e| newfront.contains(e))
            })
            .collect();

        for tgt in &out {
            let outs = self
                .circ
                .node_edges(*tgt, portgraph::graph::Direction::Outgoing);

            let ins = self
                .circ
                .node_edges(*tgt, portgraph::graph::Direction::Incoming);
            for ine in ins {
                newfront.remove(&ine);
            }
            for oute in outs {
                newfront.insert(oute);
            }
        }

        self.frontier = newfront;
        (!out.is_empty()).then_some(out)
    }
}

fn layer_dist_change(
    circ: &Circuit,
    layer: &HashSet<Vertex>,
    mapping: &Mapping,
    dists: &Distances,
    mve: &Move,
) -> i64 {
    let Move::Swap(swap_qs) = mve;

    let relevant_qs: Vec<[QAddr; 2]> = layer
        .iter()
        .filter_map(|n| {
            let ins = circ.node_edges(*n, portgraph::graph::Direction::Incoming);
            let ins_qs: Vec<_> = ins
                .into_iter()
                .map(|e| mapping.get_by_left(&e).expect("Edge missing in map"))
                .cloned()
                .collect();

            ins_qs
                .iter()
                .any(|q| swap_qs.contains(q))
                .then_some((&ins_qs[..2]).try_into().expect("Should be two qubits."))
        })
        .collect();
    let dist =
        |q1: QAddr, q2: QAddr| *dists.get(&(q1.into(), q2.into())).expect("dist missing") as i64;
    let [sq1, sq2] = swap_qs;
    relevant_qs
        .into_iter()
        .map(|rel_qs| match rel_qs {
            // TODO make this nicer
            [lq, rq] if &lq == sq1 => dist(*sq1, rq) - dist(*sq2, rq),
            [lq, rq] if &rq == sq1 => dist(*sq1, lq) - dist(*sq2, lq),
            [lq, rq] if &lq == sq2 => dist(*sq2, rq) - dist(*sq1, lq),
            [lq, rq] if &rq == sq2 => dist(*sq2, lq) - dist(*sq1, lq),
            _ => panic!("all qubits should have something do with the swap."),
        })
        .sum()
}

fn update_layer_mapping(circ: &Circuit, layer: &HashSet<Vertex>, mut mapping: Mapping) -> Mapping {
    for tgt in layer {
        let ins = circ.node_edges(*tgt, portgraph::graph::Direction::Incoming);
        let out = circ.node_edges(*tgt, portgraph::graph::Direction::Outgoing);
        for (ine, oute) in ins.into_iter().zip(out.into_iter()) {
            let mut edge_tgt = circ.edge_endpoints(oute).unwrap().1;
            let mut edge = oute;
            while {
                let op = circ.node_op(edge_tgt).unwrap();
                !(op.is_two_qb_gate() || matches!(op, Op::Output))
            } {
                (edge, edge_tgt) = next_edge(circ, edge_tgt, edge);
            }
            let (_, q) = mapping.remove_by_left(&ine).expect("missing in map");
            mapping.insert(edge, q);
        }
        // for (i, o) in ins.into_iter().zip(out.into_iter()) {
        // }

        // if circ.node_op(tgt).expect("op missing").is_two_qb_gate() {
        //     break;
        // }
        // TODO fragile, expects only 1 and 2qb gates
        // tgt = circ.edge_endpoints(o1).expect("edge missing").1;
    }

    mapping
}

#[cfg(test)]
mod tests {
    use portgraph::{
        graph::Direction,
        substitute::{BoundedSubgraph, SubgraphRef},
    };

    use crate::{
        circuit::operation::Op,
        passes::mcts::{
            distances,
            tests::{simple_arc, simple_circ},
        },
        utils::n_qbs,
        validate::check_soundness,
    };

    use super::*;

    #[test]
    fn test_insert_swap() {
        let mut c = Circuit::with_uids(n_qbs(2));
        let i = c.boundary()[0];
        let edges = c.node_edges(i, Direction::Outgoing);

        let subg = BoundedSubgraph::new(
            SubgraphRef::new(HashSet::new()),
            [edges.clone(), edges.clone()],
        );

        let rw = CircuitRewrite::new(subg, swap_circ().into(), 0.0);

        c.apply_rewrite(rw).unwrap();
        assert_eq!(c.node_count(), 3);
        check_soundness(&c).unwrap();
        // make sure incoming edges weren't deleted
        c.edge_endpoints(edges[0]).unwrap();
        c.edge_endpoints(edges[1]).unwrap();
    }

    #[test]
    fn test_advance_frontier() {
        let mut circ = Circuit::with_uids(n_qbs(3));
        // TODO add RzF64 + ZZphase gate
        circ.append_op(Op::H, &[0]).unwrap();
        let cx01 = circ.append_op(Op::CX, &[0, 1]).unwrap();
        let cx12 = circ.append_op(Op::CX, &[1, 2]).unwrap();

        let frontier = circ.node_edges(circ.boundary()[0], Direction::Outgoing);
        let mut mapping = Mapping::from_iter([
            (frontier[0], QAddr(0)),
            (frontier[1], QAddr(3)),
            (frontier[2], QAddr(2)),
        ]);

        let arc = simple_arc();
        let gates_traversed = advance_frontier(&circ, &arc, &mut mapping);
        assert_eq!(gates_traversed, 1);

        let cx12in = circ.node_edges(cx12, Direction::Incoming);
        let cx01out = circ.node_edges(cx01, Direction::Outgoing);
        let correct_mapping = Mapping::from_iter([
            (cx01out[0], QAddr(0)),
            (cx12in[0], QAddr(3)),
            (frontier[2], QAddr(2)),
        ]);

        assert_eq!(correct_mapping, mapping);
    }

    #[test]
    fn test_swap_gen() {
        let circ = simple_circ();
        let frontier = circ.node_edges(circ.boundary()[0], Direction::Outgoing);
        let mapping = Mapping::from_iter([
            (frontier[0], QAddr(0)),
            (frontier[1], QAddr(2)),
            (frontier[2], QAddr(3)),
        ]);
        let ns = NodeState {
            circ: circ.clone(),
            mapping,
        };
        let mut swaps: Vec<_> = ns.all_moves(&simple_arc()).collect();

        // only 0, 3 can be swapped because 2 is disconnected
        assert_eq!(swaps.len(), 1);

        let mut newc = circ;
        newc.apply_rewrite(ns.gen_rewrite(swaps.remove(0).0))
            .unwrap();
        check_soundness(&newc).unwrap();
    }

    #[test]
    fn test_dist_heuristic() {
        let mut circ = Circuit::with_uids(n_qbs(4));
        let cx1 = circ.append_op(Op::CX, &[1, 3]).unwrap();
        let cx2 = circ.append_op(Op::CX, &[2, 0]).unwrap();
        circ.append_op(Op::H, &[2]).unwrap();
        let arc = simple_arc();
        let dists = distances(&arc).unwrap();
        let frontier = circ.node_edges(circ.boundary()[0], Direction::Outgoing);
        let mapping = Mapping::from_iter([
            (frontier[0], QAddr(0)),
            (frontier[1], QAddr(3)),
            (frontier[2], QAddr(2)),
            (frontier[3], QAddr(1)),
        ]);

        let mve = Move::Swap([QAddr(0), QAddr(1)]);

        let layer = HashSet::from_iter([cx1, cx2]);

        let d_change = layer_dist_change(&circ, &layer, &mapping, &dists, &mve);

        assert_eq!(d_change, 1);

        let d_change = layer_dist_change(
            &circ,
            &layer,
            &mapping,
            &dists,
            &Move::Swap([QAddr(0), QAddr(3)]),
        );
        assert_eq!(d_change, 0);

        let d_change = layer_dist_change(
            &circ,
            &layer,
            &mapping,
            &dists,
            &Move::Swap([QAddr(4), QAddr(3)]),
        );
        assert_eq!(d_change, 0);
    }

    #[test]
    fn test_sim_heuristic() {
        let mut circ = Circuit::with_uids(n_qbs(4));
        circ.append_op(Op::CX, &[1, 3]).unwrap();

        circ.append_op(Op::CX, &[2, 0]).unwrap();
        circ.append_op(Op::CX, &[2, 1]).unwrap();
        circ.append_op(Op::H, &[0]).unwrap();
        circ.append_op(Op::CX, &[2, 0]).unwrap();
        circ.append_op(Op::H, &[2]).unwrap();

        circ.append_op(Op::CX, &[3, 1]).unwrap();
        circ.append_op(Op::CX, &[1, 3]).unwrap();

        let arc = simple_arc();
        let dists = distances(&arc).unwrap();
        let frontier = circ.node_edges(circ.boundary()[0], Direction::Outgoing);

        let mapping = Mapping::from_iter([
            (frontier[0], QAddr(0)),
            (frontier[1], QAddr(3)),
            (frontier[2], QAddr(2)),
            (frontier[3], QAddr(1)),
        ]);

        let ns = NodeState { circ, mapping };

        let val = ns.sim_heuristic(&Move::Swap([QAddr(2), QAddr(1)]), &dists, 3, 0.5);

        assert_eq!(val, 0.375);
    }

    #[test]
    fn test_twoqb_layers() {
        let mut circ = Circuit::with_uids(n_qbs(3));
        // TODO add RzF64 + ZZphase gate
        circ.append_op(Op::H, &[0]).unwrap();
        circ.append_op(Op::H, &[0]).unwrap();
        let cx1 = circ.append_op(Op::CX, &[0, 1]).unwrap();
        let cx2 = circ.append_op(Op::CX, &[1, 2]).unwrap();
        circ.append_op(Op::H, &[2]).unwrap();

        let frontier = circ.node_edges(circ.boundary()[0], Direction::Outgoing);
        let layers: Vec<_> = TwoqbLayerIter {
            frontier: frontier.into_iter().collect(),
            circ: &circ,
        }
        .collect();

        let cor_layers = vec![HashSet::from_iter([cx1]), HashSet::from_iter([cx2])];
        assert_eq!(layers, cor_layers);
    }
}
