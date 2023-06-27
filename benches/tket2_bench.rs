//use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
//use tket2::{
//    circuit::{
//        circuit::{Circuit, UnitID},
//        operation::{AngleValue, ConstValue, Op, WireType},
//    },
//    passes::{
//        apply_exhaustive,
//        classical::find_const_ops,
//        pattern::{node_equality, PatternMatcher},
//        squash::{find_singleq_rotations, squash_pattern},
//        CircFixedStructPattern,
//    },
//};
//
//fn pattern_match_bench_par(c: &mut Criterion) {
//    let qubits = vec![
//        UnitID::Qubit {
//            reg_name: "q".into(),
//            index: vec![0],
//        },
//        UnitID::Qubit {
//            reg_name: "q".into(),
//            index: vec![1],
//        },
//    ];
//    let mut pattern_circ = Circuit::with_uids(qubits.clone());
//    pattern_circ.append_op(Op::H, &[0]).unwrap();
//    pattern_circ.append_op(Op::H, &[1]).unwrap();
//    pattern_circ.append_op(Op::CX, &[0, 1]).unwrap();
//    pattern_circ.append_op(Op::H, &[0]).unwrap();
//    pattern_circ.append_op(Op::H, &[1]).unwrap();
//
//    let mut target_circ = Circuit::with_uids(qubits);
//    target_circ.append_op(Op::H, &[0]).unwrap();
//    target_circ.append_op(Op::H, &[1]).unwrap();
//
//    let mut group = c.benchmark_group("PatternMatch");
//    for i in (100..1000).step_by(100) {
//        for _ in 0..100 {
//            target_circ.append_op(Op::CX, &[0, 1]).unwrap();
//            target_circ.append_op(Op::H, &[0]).unwrap();
//            target_circ.append_op(Op::H, &[1]).unwrap();
//        }
//        let pattern = CircFixedStructPattern::from_circ(pattern_circ.clone(), node_equality());
//        let pmatcher = PatternMatcher::new(pattern, target_circ.dag_ref());
//
//        group.bench_function(BenchmarkId::new("Sequential", i), |b| {
//            b.iter(|| {
//                let ms = pmatcher.find_matches().collect::<Vec<_>>();
//                assert_eq!(ms.len(), i);
//            });
//        });
//        // group.bench_function(BenchmarkId::new("Paralllel", i), |b| {
//        //     b.iter(|| {
//        //         let ms = pmatcher.find_par_matches().collect::<Vec<_>>();
//        //         assert_eq!(ms.len(), i);
//        //     });
//        // });
//    }
//    group.finish();
//}

// fn pattern_match_bench_recurse(c: &mut Criterion) {
//     let qubits = vec![
//         UnitID::Qubit {
//             name: "q".into(),
//             index: vec![0],
//         },
//         UnitID::Qubit {
//             name: "q".into(),
//             index: vec![1],
//         },
//     ];
//     let mut pattern_circ = Circuit::with_uids(qubits.clone());
//     pattern_circ
//         .append_op(Op::H, &[0])
//         .unwrap();
//     pattern_circ
//         .append_op(Op::H, &[1])
//         .unwrap();
//     pattern_circ
//         .append_op(Op::CX, &vec![0, 1])
//         .unwrap();
//     pattern_circ
//         .append_op(Op::H, &[0])
//         .unwrap();
//     pattern_circ
//         .append_op(Op::H, &[1])
//         .unwrap();
//     let pattern_boundary = pattern_circ.boundary();

//     let mut target_circ = Circuit::with_uids(qubits);
//     target_circ
//         .append_op(Op::H, &[0])
//         .unwrap();
//     target_circ
//         .append_op(Op::H, &[1])
//         .unwrap();
//     target_circ
//         .append_op(Op::CX, &vec![0, 1])
//         .unwrap();
//     target_circ
//         .append_op(Op::H, &[0])
//         .unwrap();
//     target_circ
//         .append_op(Op::H, &[1])
//         .unwrap();
//     let mut group = c.benchmark_group("PatternMatch");
//     for i in (100..1000).step_by(100) {
//         for _ in 0..100 {
//             target_circ
//                 .append_op(Op::CX, &vec![0, 1])
//                 .unwrap();
//             target_circ
//                 .append_op(Op::H, &[0])
//                 .unwrap();
//             target_circ
//                 .append_op(Op::H, &[1])
//                 .unwrap();

//             pattern_circ
//                 .append_op(Op::CX, &vec![0, 1])
//                 .unwrap();
//             pattern_circ
//                 .append_op(Op::H, &[0])
//                 .unwrap();
//             pattern_circ
//                 .append_op(Op::H, &[1])
//                 .unwrap();
//         }

//         let pattern = CircFixedStructPattern::from_circ(pattern_circ, node_equality());

//         let pmatcher = PatternMatcher::new(pattern, target_circ.dag_ref());

//         group.bench_function(BenchmarkId::new("Recursive", i), |b| {
//             b.iter(|| {
//                 let ms = pmatcher.find_matches_recurse().collect::<Vec<_>>();
//                 assert_eq!(ms.len(), 1)
//             })
//         });
//         group.bench_function(BenchmarkId::new("Iterative", i), |b| {
//             b.iter(|| {
//                 let ms = pmatcher.find_matches().collect::<Vec<_>>();
//                 assert_eq!(ms.len(), 1)
//             })
//         });
//     }
//     group.finish();
// }

//fn squash_bench(c: &mut Criterion) {
//    let mut group = c.benchmark_group("Squash");
//    for size in (1..101).step_by(10) {
//        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
//            b.iter(|| {
//                let mut circ = Circuit::new();
//                // circ.add_unitid(UnitID::Qubit {
//                //     name: "q".into(),
//                //     index: vec![0],
//                // });
//                let [input, output] = circ.boundary();
//                let [mut i1, mut i2] = [(input, 0), (input, 1)];
//                for layer in 0..size {
//                    let rx = circ.add_vertex(Op::RxF64);
//                    let rz = circ.add_vertex(Op::RzF64);
//                    let cx = circ.add_vertex(Op::CX);
//                    let point5 =
//                        circ.add_vertex(Op::Const(ConstValue::Angle(AngleValue::F64(0.5))));
//                    let point2 =
//                        circ.add_vertex(Op::Const(ConstValue::Angle(AngleValue::F64(0.2))));
//                    circ.add_insert_edge(i1, (rx, 0), WireType::Qubit).unwrap();
//                    circ.add_insert_edge((rx, 0), (rz, 0), WireType::Qubit)
//                        .unwrap();
//                    circ.add_insert_edge((rz, 0), (cx, 0), WireType::Qubit)
//                        .unwrap();
//                    circ.add_insert_edge(i2, (cx, 1), WireType::Qubit).unwrap();
//                    circ.add_insert_edge((point5, 0), (rx, 1), WireType::Angle)
//                        .unwrap();
//                    circ.add_insert_edge((point2, 0), (rz, 1), WireType::Angle)
//                        .unwrap();
//
//                    circ.add_insert_edge((cx, 0), (output, layer), WireType::Qubit)
//                        .unwrap();
//
//                    i1 = (cx, 1);
//                    i2 = (input, layer + 2);
//                }
//
//                circ.add_insert_edge(i1, (output, size), WireType::Qubit)
//                    .unwrap();
//
//                let rot_replacer = |circuit| {
//                    apply_exhaustive(circuit, |c| find_singleq_rotations(c).collect()).unwrap()
//                };
//                let (circ2, success) = rot_replacer(circ);
//
//                assert!(success);
//                let squasher =
//                    |circuit| apply_exhaustive(circuit, |c| squash_pattern(c).collect()).unwrap();
//                let (circ2, success) = squasher(circ2);
//
//                assert!(success);
//
//                let constant_folder =
//                    |circuit| apply_exhaustive(circuit, |c| find_const_ops(c).collect()).unwrap();
//                let (circ2, success) = constant_folder(circ2);
//
//                assert!(success);
//                let _circ2 = circ2.remove_invalid();
//            });
//        });
//    }
//    // check_soundness(&circ2).unwrap();
//}
//
//criterion_group!(benches, squash_bench, pattern_match_bench_par);
//criterion_main!(benches);
