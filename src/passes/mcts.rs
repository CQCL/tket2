mod mctree;
mod nodestate;

use crate::circuit::circuit::{Circuit, UnitID};
use crate::circuit::dag::Edge;
use petgraph::algo::{floyd_warshall, NegativeCycle};
use petgraph::graph::{NodeIndex, UnGraph};

use std::collections::HashMap;

use self::mctree::Mcts;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
struct QAddr(pub u32);

impl From<QAddr> for NodeIndex<u32> {
    fn from(val: QAddr) -> Self {
        NodeIndex::from(val.0)
    }
}

type Architecture = UnGraph<(), ()>;
type Distances = HashMap<(NodeIndex, NodeIndex), u32>;
type Mapping = bimap::BiMap<Edge, QAddr>;

fn distances(arc: &Architecture) -> Result<Distances, NegativeCycle> {
    floyd_warshall(arc, |_| 1)
}

#[derive(Clone, Debug)]
enum Move {
    Swap([QAddr; 2]),
}

pub fn route_mcts(circ: Circuit, arc: Architecture) -> Circuit {
    let mut mcts = Mcts::new(circ, arc, 0.7);
    let res = mcts.solve();
    let s = res.state.take_nodestate().expect("unexpected child state.");
    s.circ
}

pub fn arc_from_edges(edges: impl IntoIterator<Item = (u32, u32)>) -> Architecture {
    Architecture::from_edges(edges)
}
pub fn check_mapped(circ: &Circuit, arc: &Architecture) -> Result<(), &'static str> {
    // TODO assumes qubit index refers to architecture address
    for com in circ.to_commands() {
        let qs: Vec<_> = com
            .args
            .into_iter()
            .filter_map(|q| {
                if let UnitID::Qubit { index, .. } = q {
                    Some(index[0])
                } else {
                    None
                }
            })
            .collect();

        if qs.len() != 2 {
            continue;
        }
        if !arc.contains_edge(qs[0].into(), qs[1].into()) {
            return Err("Invalid two qubit gate for architecture.");
        }
    }

    Ok(())
}
#[cfg(test)]
mod tests {
    use crate::{circuit::operation::Op, utils::n_qbs, validate::check_soundness};

    use super::*;
    fn insert_mirror_dists(dists: Distances) -> Distances {
        let mut out = HashMap::new();
        for ((i, j), d) in dists {
            if i <= j {
                out.insert((i, j), d);
                out.insert((j, i), d);
            }
        }
        out
    }
    pub(super) fn simple_circ() -> Circuit {
        let mut circ = Circuit::with_uids(n_qbs(3));
        circ.append_op(Op::CX, &[0, 1]).unwrap();
        circ.append_op(Op::CX, &[1, 2]).unwrap();
        circ
    }
    pub(super) fn simple_arc() -> Architecture {
        /*
        0 - 1 - 2
         \ / \ /
          3 - 4
        */
        Architecture::from_edges(&[(0, 1), (0, 3), (1, 2), (1, 3), (1, 4), (2, 4), (3, 4)])
    }
    #[test]
    fn test_distances() -> Result<(), NegativeCycle> {
        let g = simple_arc();

        let dists = distances(&g)?;

        let correct = HashMap::from_iter(
            [
                ((0, 1), 1),
                ((0, 2), 2),
                ((0, 3), 1),
                ((0, 4), 2),
                ((1, 2), 1),
                ((1, 3), 1),
                ((1, 4), 1),
                ((2, 4), 1),
                ((2, 3), 2),
                ((3, 4), 1),
                ((0, 0), 0),
                ((1, 1), 0),
                ((2, 2), 0),
                ((3, 3), 0),
                ((4, 4), 0),
            ]
            .map(|((i, j), d)| ((NodeIndex::new(i), NodeIndex::new(j)), d)),
        );

        let correct = insert_mirror_dists(correct);
        assert_eq!(dists, correct);
        Ok(())
    }

    #[test]
    fn test_measure_solve() {
        let mut uids = n_qbs(4);
        uids.push(UnitID::Bit {
            name: "c".into(),
            index: vec![0],
        });
        let mut circ = Circuit::with_uids(uids);
        circ.append_op(Op::CX, &[1, 3]).unwrap();

        circ.append_op(Op::CX, &[2, 0]).unwrap();
        circ.append_op(Op::CX, &[2, 1]).unwrap();
        circ.append_op(Op::H, &[0]).unwrap();
        circ.append_op(Op::CX, &[2, 0]).unwrap();
        circ.append_op(Op::H, &[2]).unwrap();
        circ.append_op(Op::Measure, &[0, 4]).unwrap();
        circ.append_op(Op::CX, &[3, 1]).unwrap();
        circ.append_op(Op::CX, &[1, 3]).unwrap();

        let arc = simple_arc();

        let outc = route_mcts(circ, arc.clone());

        check_mapped(&outc, &arc).unwrap();
        check_soundness(&outc).unwrap();
    }
}
