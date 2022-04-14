use crate::{
    circuit::{
        circuit::{Circuit, CircuitRewrite},
        dag::Vertex,
        operation::{ConstValue, Op},
    },
    graph::{graph::Direction, substitute::BoundedSubgraph},
};

pub fn find_const_ops<'c>(circ: &'c Circuit) -> impl Iterator<Item = CircuitRewrite> + '_ {
    ClRewriteIter {
        circ,
        vertex_it: circ.dag.nodes(),
    }
}

pub fn constant_fold_strat(circ: &mut Circuit) -> Result<bool, String> {
    let mut success = false;
    let mut nodes: Vec<_> = circ.dag.nodes().collect();
    loop {
        let rewrites: Vec<_> = (ClRewriteIter {
            circ,
            vertex_it: nodes.into_iter(),
        })
        .collect();
        if rewrites.is_empty() {
            break;
        }
        success = true;
        nodes = vec![];
        for rewrite in rewrites {
            for child in rewrite.graph_rewrite.subg.edges[1]
                .iter()
                .map(|e| circ.dag.edge_endpoints(*e).unwrap()[1].node)
            {
                nodes.push(child);
            }
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
                .map(|np| np.node)
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
                Op::FAdd => match &inputs[..2] {
                    [ConstValue::F64(x), ConstValue::F64(y)] => vec![ConstValue::F64(x + y)],
                    _ => return None,
                },
                Op::FMul=> match &inputs[..2] {
                    [ConstValue::F64(x), ConstValue::F64(y)] => vec![ConstValue::F64(x * y)],
                    _ => return None,
                },
                Op::FNeg => match inputs[0] {
                    ConstValue::F64(x) => vec![ConstValue::F64(-x)],
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
                    [ConstValue::F64(angle), ConstValue::F64(x), ConstValue::F64(y), ConstValue::F64(z)] => {
                        let p = -angle*std::f64::consts::PI/2.0; let s = p.sin();
                        vec![ConstValue::Quat64(cgmath::Quaternion::new(p.cos(), s*x, s*y, s*z))]
                    }
                    _ => return None
                },
                Op::QuatMul=> match &inputs[..2] {
                    [ConstValue::Quat64(x), ConstValue::Quat64(y)] => vec![ConstValue::Quat64(x * y)],
                    _ => return None,
                },
                _ => panic!("Op {:?} should not have made it to this point.", op),
            };

            let mut replace = Circuit::new();
            let [_, out] = replace.boundary();
            for (i, cv) in cvs.into_iter().enumerate() {
                let edge_type = cv.get_type();
                let cv_node = replace.add_vertex(Op::Const(cv));
                replace.add_edge((cv_node, 0), (out, i as u8), edge_type);
            }
            let subgraph = BoundedSubgraph::new(parents.into_iter().chain([n].into_iter()).into(), [vec![], self.circ.dag.node_edges(n, Direction::Outgoing).copied().collect()]);
            Some(CircuitRewrite::new(
                subgraph,
                replace.into(),
                0.0.into(),
            ))
        })
    }
}
