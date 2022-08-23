// pub mod circuit;
pub mod graph;

// pub mod json;
// pub mod passes;
// pub mod validate;

// #[cfg(test)]
// mod tests {
//     use crate::{
//         circuit::{
//             circuit::{Circuit, UnitID},
//             operation::WireType,
//             operation::{ConstValue, Op},
//         },
//         passes::{
//             apply_exhaustive, apply_greedy,
//             classical::{constant_fold_strat, find_const_ops},
//             squash::{
//                 cx_cancel_pass, find_singleq_rotations, find_singleq_rotations_pattern,
//                 squash_pattern,
//             },
//         },
//         validate::check_soundness,
//     };
//     use tket_json_rs::circuit_json::{self, SerialCircuit};

//     #[test]
//     fn read_json() {
//         // let expr = symengine::Expression::new("a + b + 3");
//         let circ_s = r#"{"bits": [["c", [0]], ["c", [1]]], "commands": [{"args": [["q", [0]]], "op": {"type": "H"}}, {"args": [["q", [0]], ["q", [1]]], "op": {"type": "CX"}}, {"args": [["q", [0]], ["c", [0]]], "op": {"type": "Measure"}}, {"args": [["q", [1]], ["c", [1]]], "op": {"type": "Measure"}}], "implicit_permutation": [[["q", [0]], ["q", [0]]], [["q", [1]], ["q", [1]]]], "phase": "0", "qubits": [["q", [0]], ["q", [1]]]}"#;
//         let ser: circuit_json::SerialCircuit = serde_json::from_str(circ_s).unwrap();
//         assert_eq!(ser.commands.len(), 4);

//         let circ: Circuit = ser.clone().into();
//         check_soundness(&circ).unwrap();

//         let _reser: SerialCircuit = circ.into();
//         assert_eq!(&ser, &_reser);

//         // ser and reser cannot be compared because they will be different up to
//         // topsort ordering of parallel commands
//     }

//     // #[test]
//     // fn test_param() {
//     //     assert_eq!(Param::new("3") + Param::new("x"), Param::new("3 + x"));
//     //     assert_eq!(Param::new("0") - Param::new("0.1"), Param::new("-0.1"));
//     //     assert_eq!(Param::new("0.1").neg(), Param::new("-0.1"));

//     //     assert!(Param::new("x").eval().is_none());
//     //     assert_eq!(Param::new("2.0 + 2.0/4").eval(), Some(2.5));
//     //     assert!(equiv_0(&Param::new("0"), 4));
//     //     assert!(equiv_0(&Param::new("4.0"), 4));
//     //     assert!(equiv_0(&Param::new("8.0"), 4));
//     //     assert!(!equiv_0(&Param::new("2.0"), 4));
//     //     assert!(equiv_0(&Param::new("2.0"), 2));
//     //     assert!(!equiv_0(&Param::new("0.5"), 2));
//     // }

//     // #[test]
//     // fn test_dagger() {
//     //     assert_eq!(Op::H.dagger().unwrap(), Op::H);
//     //     assert_eq!(Op::CX.dagger().unwrap(), Op::CX);
//     //     assert_eq!(Op::Rx(0.1.into()).dagger().unwrap(), Op::Rx((-0.1).into()));
//     //     assert_eq!(
//     //         Op::Rz(Param::new("x")).dagger().unwrap(),
//     //         Op::Rz(Param::new("-x"))
//     //     );
//     // }

//     #[test]
//     fn test_fadd() {
//         let mut circ = Circuit::new();

//         circ.add_unitid(UnitID::F64("a".into()));
//         circ.add_unitid(UnitID::F64("b".into()));
//         let [input, output] = circ.boundary();

//         let fadd = circ.add_vertex(Op::AngleAdd);
//         circ.tup_add_edge((input, 0), (fadd, 0), WireType::F64);

//         circ.tup_add_edge((input, 1), (fadd, 1), WireType::F64);

//         circ.tup_add_edge((fadd, 0), (output, 0), WireType::F64);
//     }

//     #[test]
//     fn test_copy() {
//         let mut circ = Circuit::new();

//         circ.add_unitid(UnitID::F64("a".into()));
//         let [input, output] = circ.boundary();

//         let fadd = circ.add_vertex(Op::AngleAdd);
//         let e = circ.tup_add_edge((input, 0), (fadd, 0), WireType::F64);

//         let copy = circ.copy_edge(e, 2).unwrap();

//         circ.tup_add_edge((copy, 1), (fadd, 1), WireType::F64);

//         circ.tup_add_edge((fadd, 0), (output, 0), WireType::F64);
//         // println!("{}", dot_string(&circ.dag));
//     }

//     #[test]
//     fn test_const_fold_simple() {
//         let mut circ = Circuit::new();

//         let [_, output] = circ.boundary();

//         let fadd = circ.add_vertex(Op::AngleAdd);
//         let one = circ.add_vertex(Op::Const(ConstValue::f64_angle(0.5)));
//         let two = circ.add_vertex(Op::Const(ConstValue::f64_angle(1.5)));
//         let _e1 = circ.tup_add_edge((one, 0), (fadd, 0), WireType::Angle);
//         let _e2 = circ.tup_add_edge((two, 0), (fadd, 1), WireType::Angle);

//         let _out = circ.tup_add_edge((fadd, 0), (output, 0), WireType::Angle);
//         check_soundness(&circ).unwrap();

//         let rewrite = find_const_ops(&circ).next().unwrap();

//         circ.apply_rewrite(rewrite).unwrap();

//         // println!("{}", dot_string(&circ.dag));
//         assert_eq!(circ.dag.node_count(), 3);
//         assert_eq!(circ.dag.edge_count(), 1);
//         let mut nodeit = circ.dag.node_weights();
//         // skip input and output
//         nodeit.next();
//         nodeit.next();

//         assert_eq!(
//             &nodeit.next().unwrap().op,
//             &Op::Const(ConstValue::f64_angle(2.0))
//         );

//         check_soundness(&circ).unwrap();
//     }

//     #[test]
//     fn test_const_fold_less_simple() {
//         let mut circ = Circuit::new();
//         circ.add_unitid(UnitID::Qubit {
//             reg_name: "q".into(),
//             index: vec![0],
//         });

//         let [input, output] = circ.boundary();

//         // Rx(8.0 + (-(2.0)) + 0.5 + 0.5) q[0]
//         // note 0.5 copied
//         let fadd1 = circ.add_vertex(Op::AngleAdd);
//         let fadd2 = circ.add_vertex(Op::AngleAdd);
//         let fadd3 = circ.add_vertex(Op::AngleAdd);
//         let neg = circ.add_vertex(Op::AngleNeg);
//         let copy = circ.add_vertex(Op::Copy {
//             n_copies: 2,
//             typ: WireType::Angle,
//         });

//         let rx = circ.add_vertex(Op::RxF64);
//         circ.tup_add_edge((input, 0), (rx, 0), WireType::Qubit);

//         let point5 = circ.add_vertex(Op::Const(ConstValue::f64_angle(0.5)));
//         let two = circ.add_vertex(Op::Const(ConstValue::f64_angle(2.0)));
//         let eight = circ.add_vertex(Op::Const(ConstValue::f64_angle(8.0)));

//         circ.tup_add_edge((two, 0), (neg, 0), WireType::Angle);

//         circ.tup_add_edge((neg, 0), (fadd1, 0), WireType::Angle);

//         circ.tup_add_edge((eight, 0), (fadd1, 1), WireType::Angle);

//         circ.tup_add_edge((point5, 0), (copy, 0), WireType::Angle);

//         circ.tup_add_edge((copy, 0), (fadd3, 0), WireType::Angle);
//         circ.tup_add_edge((copy, 1), (fadd2, 0), WireType::Angle);

//         circ.tup_add_edge((fadd1, 0), (fadd2, 1), WireType::Angle);

//         circ.tup_add_edge((fadd2, 0), (fadd3, 1), WireType::Angle);

//         circ.tup_add_edge((fadd3, 0), (rx, 1), WireType::Angle);

//         circ.tup_add_edge((rx, 0), (output, 0), WireType::Qubit);

//         assert_eq!(circ.dag.node_count(), 11);
//         assert_eq!(circ.dag.edge_count(), 11);
//         check_soundness(&circ).unwrap();

//         let orig_circ = circ.clone();
//         let mut orig_circ2 = circ.clone();
//         let rewrites: Vec<_> = find_const_ops(&circ).collect();

//         assert_eq!(rewrites.len(), 2);

//         for rewrite in rewrites {
//             circ.apply_rewrite(rewrite).unwrap();
//         }
//         check_soundness(&circ).unwrap();

//         assert_eq!(circ.dag.node_count(), 10);
//         assert_eq!(circ.dag.edge_count(), 9);

//         assert_eq!(
//             circ.dag
//                 .node_weights()
//                 .filter(|v| matches!(v.op, Op::Const(_)))
//                 .count(),
//             4
//         );

//         assert_eq!(
//             circ.dag
//                 .node_weights()
//                 .filter(|op| matches!(op.op, Op::AngleNeg))
//                 .count(),
//             0
//         );

//         assert_eq!(
//             circ.dag
//                 .node_weights()
//                 .filter(|op| matches!(op.op, Op::AngleAdd))
//                 .count(),
//             3
//         );

//         // evaluate all the additions
//         for _ in 0..3 {
//             let rewrites: Vec<_> = find_const_ops(&circ).collect();

//             assert_eq!(rewrites.len(), 1);

//             circ.apply_rewrite(rewrites.into_iter().next().unwrap())
//                 .unwrap();
//         }
//         check_soundness(&circ).unwrap();

//         let constant_folder =
//             |circuit| apply_exhaustive(circuit, |c| find_const_ops(c).collect()).unwrap();

//         // the above should replicate doing it all in one go
//         let (circ2, success) = constant_folder(orig_circ);
//         check_soundness(&circ2).unwrap();

//         assert!(success);

//         let (circ, success) = constant_folder(circ);
//         check_soundness(&circ).unwrap();

//         assert!(!success);

//         assert!(constant_fold_strat(&mut orig_circ2).unwrap());

//         for c in [circ, circ2, orig_circ2] {
//             let c = c.remove_invalid();
//             assert_eq!(c.dag.node_count(), 4);
//             assert_eq!(c.dag.edge_count(), 3);
//             let const_val = c
//                 .dag
//                 .node_weights()
//                 .find_map(|v| {
//                     if let Op::Const(x) = &v.op {
//                         Some(x)
//                     } else {
//                         None
//                     }
//                 })
//                 .unwrap();

//             assert_eq!(const_val, &ConstValue::f64_angle(7.0));
//         }
//     }

//     #[test]
//     fn test_rotation_replace() {
//         let mut circ = Circuit::new();
//         circ.add_unitid(UnitID::Qubit {
//             reg_name: "q".into(),
//             index: vec![0],
//         });
//         let [input, output] = circ.boundary();

//         let point5 = circ.add_vertex(Op::Const(ConstValue::f64_angle(0.5)));
//         let rx = circ.add_vertex(Op::RxF64);
//         circ.tup_add_edge((input, 0), (rx, 0), WireType::Qubit);
//         circ.tup_add_edge((point5, 0), (rx, 1), WireType::Angle);
//         circ.tup_add_edge((rx, 0), (output, 0), WireType::Qubit);
//         check_soundness(&circ).unwrap();

//         // let rot_replacer =
//         // |circuit| apply_exhaustive(circuit, |c| find_singleq_rotations(c).collect()).unwrap();

//         let rot_replacer = |circuit| {
//             apply_exhaustive(circuit, |c| find_singleq_rotations_pattern(c).collect()).unwrap()
//         };
//         let (circ2, success) = rot_replacer(circ);
//         check_soundness(&circ2).unwrap();

//         assert!(success);

//         let constant_folder =
//             |circuit| apply_exhaustive(circuit, |c| find_const_ops(c).collect()).unwrap();
//         let (circ2, success) = constant_folder(circ2);
//         check_soundness(&circ2).unwrap();
//         assert!(success);
//         // use crate::graph::dot::dot_string;
//         // println!("{}", dot_string(&_circ2.dag));
//     }

//     #[test]
//     fn test_squash() {
//         let mut circ = Circuit::new();
//         circ.add_unitid(UnitID::Qubit {
//             reg_name: "q".into(),
//             index: vec![0],
//         });
//         let [input, output] = circ.boundary();

//         let rx = circ.add_vertex(Op::RxF64);
//         let rz = circ.add_vertex(Op::RzF64);
//         circ.tup_add_edge((input, 0), (rx, 0), WireType::Qubit);
//         circ.tup_add_edge((input, 1), (rx, 1), WireType::Angle);
//         circ.tup_add_edge((rx, 0), (rz, 0), WireType::Qubit);
//         circ.tup_add_edge((input, 2), (rz, 1), WireType::Angle);
//         circ.tup_add_edge((rz, 0), (output, 0), WireType::Qubit);
//         check_soundness(&circ).unwrap();

//         let rot_replacer =
//             |circuit| apply_exhaustive(circuit, |c| find_singleq_rotations(c).collect()).unwrap();
//         let (circ2, success) = rot_replacer(circ);

//         check_soundness(&circ2).unwrap();
//         assert!(success);
//         // let squasher =
//         // |circuit| apply_exhaustive(circuit, |c| SquashFindIter::new(c).collect()).unwrap();
//         let squasher = |circuit| apply_greedy(circuit, |c| squash_pattern(c).next()).unwrap();
//         let (mut circ2, success) = squasher(circ2);

//         assert!(success);
//         check_soundness(&circ2).unwrap();

//         circ2
//             .bind_input(PortIndex::new(1), ConstValue::f64_angle(0.5))
//             .unwrap();
//         circ2
//             .bind_input(PortIndex::new(2), ConstValue::f64_angle(0.2))
//             .unwrap();

//         // use crate::graph::dot::dot_string;
//         // println!("{}", dot_string(&circ2.dag));
//         let constant_folder =
//             |circuit| apply_exhaustive(circuit, |c| find_const_ops(c).collect()).unwrap();
//         let (circ2, success) = constant_folder(circ2);

//         assert!(success);
//         check_soundness(&circ2).unwrap();

//         // let _circ2 = _circ2.remove_invalid();
//         // use crate::graph::dot::dot_string;
//         // println!("{}", dot_string(&_circ2.dag));
//         // TODO verify behaviour at each step
//     }

//     #[test]
//     fn test_cx_cancel() {
//         let qubits = vec![
//             UnitID::Qubit {
//                 reg_name: "q".into(),
//                 index: vec![0],
//             },
//             UnitID::Qubit {
//                 reg_name: "q".into(),
//                 index: vec![1],
//             },
//         ];
//         let mut circ = Circuit::with_uids(qubits);
//         circ.append_op(Op::CX, &vec![PortIndex::new(0), PortIndex::new(1)])
//             .unwrap();
//         circ.append_op(Op::CX, &vec![PortIndex::new(0), PortIndex::new(1)])
//             .unwrap();
//         check_soundness(&circ).unwrap();

//         let (circ, success) = cx_cancel_pass(circ);
//         assert!(success);
//         check_soundness(&circ).unwrap();

//         assert_eq!(circ.dag.node_count(), 2);
//     }
// }
