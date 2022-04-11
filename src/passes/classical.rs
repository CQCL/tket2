use crate::{
    circuit::{
        circuit::Circuit,
        dag::{CircuitRewrite, Vertex},
        operation::{ConstValue, Op},
    },
    graph::{
        graph::Direction,
        substitute::{BoundedGraph, Cut},
    },
};

pub fn find_const_ops<'c>(
    circ: &'c Circuit,
) -> ClRewriteIter<'c, impl Iterator<Item = Vertex> + 'c> {
    ClRewriteIter {
        circ,
        vertex_it: circ.dag.nodes(),
    }
}
pub struct ClRewriteIter<'c, I: Iterator<Item = Vertex>> {
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
                Op::FNeg => match inputs[0] {
                    ConstValue::F64(x) => vec![ConstValue::F64(-x)],
                    _ => return None,
                },
                Op::Copy { n_copies, .. } => vec![inputs[0].clone(); *n_copies as usize],
                _ => panic!("Op {:?} should not have made it to this point.", op),
            };

            let mut replace = Circuit::new();
            let [inp, out] = replace.boundary();
            for (i, cv) in cvs.into_iter().enumerate() {
                let edge_type = cv.get_type();
                let cv_node = replace.add_vertex(Op::Const(cv));
                replace.add_edge((cv_node, 0), (out, i as u8), edge_type);
            }
            Some(CircuitRewrite::new(
                Cut::new(parents, vec![n]),
                BoundedGraph {
                    graph: replace.dag,
                    entry: inp,
                    exit: out,
                },
            ))
        })
    }
}

/// Repeatedly apply all available constant folding rewrites until no more are found.
///
/// # Errors
///
/// This function will return an error if rewrite application fails.
pub fn const_fold_exhaustive(mut circ: Circuit) -> Result<(Circuit, bool), String> {
    let mut success = false;
    loop {
        let rewrites: Vec<_> = find_const_ops(&circ).collect();
        if rewrites.is_empty() {
            break;
        }
        success = true;
        for rewrite in rewrites {
            circ.dag.apply_rewrite(rewrite)?;
        }
    }

    Ok((circ, success))
}
