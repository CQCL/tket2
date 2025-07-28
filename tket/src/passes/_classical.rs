use std::collections::BTreeSet;

use crate::circuit::{
    circuit::{Circuit, CircuitRewrite},
    dag::Vertex,
    operation::{ConstValue, Op, Quat},
};
use portgraph::{
    graph::Direction,
    substitute::{BoundedSubgraph, RewriteError, SubgraphRef},
};

pub fn find_const_ops(circ: &Circuit) -> impl Iterator<Item = CircuitRewrite> + '_ {
    ClRewriteIter {
        circ,
        vertex_it: circ.dag.node_indices(),
    }
}

pub fn constant_fold_strat(circ: &mut Circuit) -> Result<bool, RewriteError> {
    let mut success = false;
    let mut nodes: BTreeSet<_> = circ.dag.node_indices().collect();
    loop {
        let rewrites: Vec<_> = (ClRewriteIter {
            circ,
            vertex_it: nodes.iter().copied(),
        })
        .collect();
        if rewrites.is_empty() {
            break;
        }
        success = true;
        nodes.clear();
        for rewrite in rewrites {
            nodes.extend(
                rewrite.graph_rewrite.subg.edges[1]
                    .iter()
                    .map(|e| circ.dag.edge_endpoint(*e, Direction::Incoming).unwrap()),
            );
            circ.apply_rewrite(rewrite)?;
        }
    }

    Ok(success)
}
struct ClRewriteIter<'c, I: Iterator<Item = Vertex>> {
    circ: &'c Circuit,
    vertex_it: I,
}

impl<'circ, I: Iterator<Item = Vertex>> Iterator for ClRewriteIter<'circ, I> {
    type Item = CircuitRewrite;

    fn next(&mut self) -> Option<Self::Item> {
        self.vertex_it.find_map(|n| {
            let op = &self.circ.dag.node_weight(n).unwrap().op;
            if !op.is_pure_classical() || matches!(op, Op::Const(_)) {
                return None;
            }

            let parents: Vec<_> = self
                .circ
                .dag
                .neighbours(n, Direction::Incoming)
                .collect();

            let inputs: Option<Vec<_>> = parents
                .iter()
                .map(|n| match &self.circ.dag.node_weight(*n).unwrap().op {
                    Op::Const(c) => Some(c),
                    _ => None,
                })
                .collect();
            let inputs = inputs?;

            let cvs = match op {
                Op::AngleAdd => match &inputs[..2] {
                    [ConstValue::Angle(x), ConstValue::Angle(y)] => vec![ConstValue::Angle(x + y)],
                    _ => return None,
                },
                Op::AngleMul=> match &inputs[..2] {
                    [ConstValue::Angle(x), ConstValue::Angle(y)] => vec![ConstValue::Angle(x * y)],
                    _ => return None,
                },
                Op::AngleNeg => match inputs[0] {
                    ConstValue::Angle(x) => vec![ConstValue::Angle(-x)],
                    _ => return None,
                },

                // Op::Sin => match inputs[0] {
                //     ConstValue::F64(x) => vec![ConstValue::F64(x.sin())],
                //     _ => return None,
                // },
                // Op::Cos => match inputs[0] {
                //     ConstValue::F64(x) => vec![ConstValue::F64(x.cos())],
                //     _ => return None,
                // },
                Op::Copy { n_copies, .. } => vec![inputs[0].clone(); *n_copies as usize],
                Op::ToRotation => match &inputs[..4] {
                    [ConstValue::Angle(angle), ConstValue::F64(x), ConstValue::F64(y), ConstValue::F64(z)] => {
                        let p = -angle.radians()/2.0;
                        let s = p.sin();
                        vec![ConstValue::Quat64(Quat(cgmath::Quaternion::new(p.cos(), s*x, s*y, s*z)))]
                    }
                    _ => return None
                },
                Op::QuatMul=> match &inputs[..2] {
                    [ConstValue::Quat64(x), ConstValue::Quat64(y)] => vec![ConstValue::Quat64(Quat(x.0 * y.0))],
                    _ => return None,
                },
                _ => panic!("Op {op:?} should not have made it to this point."),
            };

            let mut replace = Circuit::new();
            for cv in cvs.into_iter() {
                // let edge_type = cv.get_type();
                let e = replace.new_output(cv.get_type());
                replace.add_vertex_with_edges(Op::Const(cv), vec![], vec![e]);

                // replace.tup_add_edge((cv_node, 0), (out, i as u8), edge_type);
            }
            let subgraph = BoundedSubgraph::new(
                SubgraphRef::from_iter(parents
                        .into_iter()
                        .chain([n].into_iter())
                        ),
                [vec![], self.circ.dag.node_edges(n, Direction::Outgoing).collect()]
            );
            Some(CircuitRewrite::new(
                subgraph,
                replace.into(),
                0.0,
            ))
        })
    }
}
