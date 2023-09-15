use crate::circuit::{
    circuit::UnitID,
    operation::{ConstValue, WireType},
};

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

    // permute addition operations
    let rx = circ.add_vertex(Op::RzF64);
    let point5 = circ.add_vertex(Op::Const(ConstValue::f64_angle(0.5)));
    circ.add_insert_edge((rx, 0), (output, 0), WireType::Qubit)
        .unwrap();
    circ.add_insert_edge((input, 0), (rx, 0), WireType::Qubit)
        .unwrap();
    circ.add_insert_edge((point5, 0), (rx, 1), WireType::Angle)
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

#[test]
fn test_taso_small() {
    // Figure 6 from Quartz paper https://arxiv.org/pdf/2204.09033.pdf

    let repsets = small_repset();

    test_taso(repsets);
}

fn test_taso(repsets: Vec<RepCircSet>) {
    let circ = sample_circ();
    let mut correct = Circuit::with_uids(n_qbs(4));
    correct.append_op(Op::H, &[0]).unwrap();
    correct.append_op(Op::H, &[3]).unwrap();
    correct.append_op(Op::CX, &[3, 2]).unwrap();
    correct.append_op(Op::CX, &[2, 1]).unwrap();
    correct.append_op(Op::CX, &[1, 0]).unwrap();
    for f in [taso, |c, ps, g, cst, tmo| taso_mpsc(c, ps, g, cst, tmo, 4)] {
        let cout = f(circ.clone(), repsets.clone(), 1.2, Circuit::node_count, 10);

        assert_ne!(circuit_hash(&circ), circuit_hash(&cout));
        assert_eq!(circuit_hash(&correct), circuit_hash(&cout));
    }
}

fn small_repset() -> Vec<RepCircSet> {
    let mut two_h = Circuit::with_uids(n_qbs(1));
    two_h.append_op(Op::H, &[0]).unwrap();
    two_h.append_op(Op::H, &[0]).unwrap();

    let oneqb_id = Circuit::with_uids(n_qbs(1));

    let mut cx_left = Circuit::with_uids(vec![
        UnitID::Qubit {
            reg_name: "q".into(),
            index: vec![0],
        },
        UnitID::Qubit {
            reg_name: "q".into(),
            index: vec![1],
        },
    ]);
    cx_left.append_op(Op::CX, &[0, 1]).unwrap();
    cx_left.append_op(Op::H, &[0]).unwrap();
    cx_left.append_op(Op::H, &[1]).unwrap();

    let mut cx_right = Circuit::with_uids(vec![
        UnitID::Qubit {
            reg_name: "q".into(),
            index: vec![0],
        },
        UnitID::Qubit {
            reg_name: "q".into(),
            index: vec![1],
        },
    ]);
    cx_right.append_op(Op::H, &[0]).unwrap();
    cx_right.append_op(Op::H, &[1]).unwrap();
    cx_right.append_op(Op::CX, &[1, 0]).unwrap();

    let repsets = vec![
        RepCircSet {
            rep_circ: oneqb_id,
            others: vec![two_h],
        },
        RepCircSet {
            rep_circ: cx_left,
            others: vec![cx_right],
        },
    ];
    repsets
}

fn n_qbs(n: u32) -> Vec<UnitID> {
    (0..n)
        .map(|i| UnitID::Qubit {
            reg_name: "q".into(),
            index: vec![i],
        })
        .collect()
}

fn sample_circ() -> Circuit {
    let mut circ = Circuit::with_uids(n_qbs(4));
    circ.append_op(Op::H, &[1]).unwrap();
    circ.append_op(Op::H, &[2]).unwrap();
    circ.append_op(Op::CX, &[2, 3]).unwrap();
    circ.append_op(Op::H, &[3]).unwrap();
    circ.append_op(Op::CX, &[1, 2]).unwrap();
    circ.append_op(Op::H, &[2]).unwrap();
    circ.append_op(Op::CX, &[0, 1]).unwrap();
    circ.append_op(Op::H, &[1]).unwrap();
    circ.append_op(Op::H, &[0]).unwrap();
    circ
}

#[test]
fn test_taso_big() {
    let repsets = rep_sets_from_path("test_files/h_rz_cxcomplete_ECC_set.json");
    test_taso(repsets);
}
