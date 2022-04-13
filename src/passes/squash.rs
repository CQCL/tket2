use crate::{
    circuit::{
        circuit::{Circuit, CircuitRewrite},
        dag::Vertex,
        operation::{ConstValue, Op, WireType},
    },
    graph::substitute::Cut,
};

pub fn find_singleq_rotations<'c>(circ: &'c Circuit) -> impl Iterator<Item = CircuitRewrite> + '_ {
    RotationRewriteIter {
        circ,
        vertex_it: circ.dag.nodes(),
    }
}
pub struct RotationRewriteIter<'c, I: Iterator<Item = Vertex>> {
    circ: &'c Circuit,
    vertex_it: I,
}

// fn add_div2(circ: &mut Circuit, inp: Vertex) -> Vertex {
//     let neg = circ.add_vertex(Op::FNeg);
//     let div2 = circ.add_vertex(Op::FMul);
//     let point5 = circ.add_vertex(Op::Const(ConstValue::F64(0.5)));
//     circ.add_edge((inp, 1), (neg, 0), WireType::F64);
//     circ.add_edge((point5, 0), (div2, 0), WireType::F64);
//     circ.add_edge((neg, 0), (div2, 1), WireType::F64);

//     div2
// }

// fn unary_op_f64(circ: &mut Circuit, inp: NodePort, op: Op) -> Vertex {
//     let v = circ.add_vertex(op);
//     circ.add_edge(inp, (v, 0).into(), WireType::F64);
//     v
// }

impl<'circ, I: Iterator<Item = Vertex>> Iterator for RotationRewriteIter<'circ, I> {
    type Item = CircuitRewrite;

    fn next(&mut self) -> Option<Self::Item> {
        self.vertex_it.find_map(|n| {
            let op = &self.circ.dag.node_weight(n).unwrap().op;
            if !op.is_one_qb_gate() || matches!(op, Op::Rotation) {
                return None;
            }

            let mut replace = Circuit::new();
            let [inp, out] = replace.boundary();
            let make_quat = replace.add_vertex(Op::ToRotation);
            let rot = replace.add_vertex(Op::Rotation);

            replace.add_edge((inp, 0), (rot, 0), WireType::Qubit);
            replace.add_edge((make_quat, 0), (rot, 1), WireType::Quat64);

            let incoming_ports = match op {
                Op::RxF64 => {
                    // replace.add_edge((inp, 1), (make_quat, 0), WireType::F64);
                    let x = replace.add_vertex(Op::Const(ConstValue::F64(1.0)));
                    let y = replace.add_vertex(Op::Const(ConstValue::F64(0.0)));
                    let z = replace.add_vertex(Op::Const(ConstValue::F64(0.0)));
                    [(inp, 1), (x, 0), (y, 0), (z, 0)]
                }
                Op::RzF64 => {
                    // replace.add_edge((inp, 1), (make_quat, 0), WireType::F64);
                    let x = replace.add_vertex(Op::Const(ConstValue::F64(0.0)));
                    let y = replace.add_vertex(Op::Const(ConstValue::F64(0.0)));
                    let z = replace.add_vertex(Op::Const(ConstValue::F64(1.0)));
                    [(inp, 1), (x, 0), (y, 0), (z, 0)]
                }

                _ => panic!("Op {:?} should not have made it to this point.", op),
            };
            for (port_index, source) in incoming_ports.into_iter().enumerate() {
                replace.add_edge(source, (make_quat, port_index as u8), WireType::F64);
            }
            replace.add_edge((rot, 0), (out, 0), WireType::Qubit);
            Some(CircuitRewrite::new(
                Cut::new(vec![n], vec![n]),
                replace.into(),
                0.0.into(),
            ))
        })
    }
}
