use std::collections::HashMap;

use portgraph::{
    graph::{Direction, NodeIndex},
    toposort::TopSortWalker,
};

use crate::circuit::{circuit::Circuit, operation::Op};

// type Transformation = (Circuit, Circuit);

// pub fn taso<C>(
//     circ: Circuit,
//     transforms: Vec<Transformation>,
//     gamma: f64,
//     cost: C,
//     timeout: i64,
// ) -> Circuit
// where
//     C: Fn(&Circuit) -> f64,
// {
//     todo!()
// }

#[allow(dead_code)]
fn op_hash(op: &Op) -> Option<usize> {
    Some(match op {
        Op::H => 1,
        Op::CX => 2,
        Op::ZZMax => 3,
        Op::Reset => 4,
        Op::Input => 5,
        Op::Output => 6,
        Op::Noop(_) => 7,
        Op::Measure => 8,
        Op::Barrier => 9,
        Op::AngleAdd => 10,
        Op::AngleMul => 11,
        Op::AngleNeg => 12,
        Op::QuatMul => 13,
        // Op::Copy { n_copies, typ } => todo!(),
        Op::RxF64 => 14,
        Op::RzF64 => 15,
        Op::TK1 => 16,
        Op::Rotation => 17,
        Op::ToRotation => 18,
        // should Const of different values be hash different?
        Op::Const(_) => 19,
        // Op::Custom(_) => todo!(),
        _ => return None,
    })
}

#[allow(dead_code)]
fn circuit_hash(circ: &Circuit) -> usize {
    // adapted from Quartz (Apache 2.0)
    // https://github.com/quantum-compiler/quartz/blob/2e13eb7ffb3c5c5fe96cf5b4246f4fd7512e111e/src/quartz/tasograph/tasograph.cpp#L410
    let mut total = 0;

    let mut hash_vals: HashMap<NodeIndex, usize> = HashMap::new();
    let [i, _] = circ.boundary();

    hash_vals.insert(i, 17 * 13 + op_hash(&Op::Input).expect("unhashable op"));

    for nid in TopSortWalker::new(
        circ.dag_ref(),
        circ.dag
            .node_indices()
            .filter(|n| circ.dag.node_edges(*n, Direction::Incoming).count() == 0)
            .collect(),
    ) {
        if hash_vals.contains_key(&nid) {
            continue;
        }

        let mut myhash =
            17 * 13 + op_hash(circ.node_op(nid).expect("op not found.")).expect("unhashable op");

        for ine in circ.node_edges(nid, Direction::Incoming) {
            let (src, _) = circ.edge_endpoints(ine).expect("edge not found.");
            debug_assert!(hash_vals.contains_key(&src));

            let mut edgehash = hash_vals[&src];

            edgehash = edgehash * 31
                + (circ
                    .port_of_edge(src, ine, Direction::Outgoing)
                    .expect("edge not found."));
            edgehash = edgehash * 31
                + (circ
                    .port_of_edge(nid, ine, Direction::Incoming)
                    .expect("edge not found."));

            myhash += edgehash
        }

        hash_vals.insert(nid, myhash);
        total += myhash;
    }

    total
}

#[cfg(test)]
mod tests {
    use crate::circuit::operation::{ConstValue, WireType};

    use super::*;

    #[test]
    fn test_simple_hash() {
        let mut circ1 = Circuit::new();
        let [input, output] = circ1.boundary();

        let point5 = circ1.add_vertex(Op::Const(ConstValue::f64_angle(0.5)));
        let rx = circ1.add_vertex(Op::RzF64);
        circ1
            .add_insert_edge((input, 0), (rx, 0), WireType::Qubit)
            .unwrap();
        circ1
            .add_insert_edge((point5, 0), (rx, 1), WireType::Angle)
            .unwrap();
        circ1
            .add_insert_edge((rx, 0), (output, 0), WireType::Qubit)
            .unwrap();

        let mut circ = Circuit::new();
        let [input, output] = circ.boundary();

        let point5 = circ.add_vertex(Op::Const(ConstValue::f64_angle(0.5)));
        let rx = circ.add_vertex(Op::RzF64);
        circ.add_insert_edge((input, 0), (rx, 0), WireType::Qubit)
            .unwrap();
        circ.add_insert_edge((point5, 0), (rx, 1), WireType::Angle)
            .unwrap();
        circ.add_insert_edge((rx, 0), (output, 0), WireType::Qubit)
            .unwrap();

        assert_eq!(circuit_hash(&circ), circuit_hash(&circ1));

        let mut circ = Circuit::new();
        let [input, output] = circ.boundary();

        let point5 = circ.add_vertex(Op::Const(ConstValue::f64_angle(0.5)));
        let rx = circ.add_vertex(Op::RxF64); // rx rather than rz
        circ.add_insert_edge((input, 0), (rx, 0), WireType::Qubit)
            .unwrap();
        circ.add_insert_edge((point5, 0), (rx, 1), WireType::Angle)
            .unwrap();
        circ.add_insert_edge((rx, 0), (output, 0), WireType::Qubit)
            .unwrap();

        assert_ne!(circuit_hash(&circ), circuit_hash(&circ1));
    }

    // TODO test that takes a list of circuits (some of them very related to
    // each other but distinct) and checks for no hash collisions
}
