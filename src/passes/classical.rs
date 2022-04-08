use crate::{
    circuit::{
        circuit::Circuit,
        dag::{CircuitRewrite, Vertex},
        operation::{ConstValue, Op},
    },
    graph::{
        graph::{Direction, NodePort, PortIndex},
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
            match op {
                Op::FAdd | Op::FNeg | Op::Copy { .. } => (),
                _ => return None,
            };
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
                replace.add_edge(
                    NodePort::new(cv_node, PortIndex::new(0)),
                    NodePort::new(out, PortIndex::new(i)),
                    edge_type,
                );
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
