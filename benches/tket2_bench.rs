use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use rayon::iter::ParallelIterator;
use tket_rs::{
    circuit::{
        circuit::{Circuit, UnitID},
        operation::Op,
    },
    graph::graph::PortIndex,
    passes::{
        pattern::{node_equality, FixedStructPattern, PatternMatcher},
        CircFixedStructPattern,
    },
};

fn pattern_match_bench_par(c: &mut Criterion) {
    let qubits = vec![
        UnitID::Qubit {
            name: "q".into(),
            index: vec![0],
        },
        UnitID::Qubit {
            name: "q".into(),
            index: vec![1],
        },
    ];
    let mut pattern_circ = Circuit::with_uids(qubits.clone());
    pattern_circ
        .append_op(Op::H, &vec![PortIndex::new(0)])
        .unwrap();
    pattern_circ
        .append_op(Op::H, &vec![PortIndex::new(1)])
        .unwrap();
    pattern_circ
        .append_op(Op::CX, &vec![PortIndex::new(0), PortIndex::new(1)])
        .unwrap();
    pattern_circ
        .append_op(Op::H, &vec![PortIndex::new(0)])
        .unwrap();
    pattern_circ
        .append_op(Op::H, &vec![PortIndex::new(1)])
        .unwrap();
    let pattern_boundary = pattern_circ.boundary();

    let mut target_circ = Circuit::with_uids(qubits);
    target_circ
        .append_op(Op::H, &vec![PortIndex::new(0)])
        .unwrap();
    target_circ
        .append_op(Op::H, &vec![PortIndex::new(1)])
        .unwrap();

    let mut group = c.benchmark_group("PatternMatch");
    for i in (100..2000).step_by(100) {
        for _ in 0..100 {
            target_circ
                .append_op(Op::CX, &vec![PortIndex::new(0), PortIndex::new(1)])
                .unwrap();
            target_circ
                .append_op(Op::H, &vec![PortIndex::new(0)])
                .unwrap();
            target_circ
                .append_op(Op::H, &vec![PortIndex::new(1)])
                .unwrap();
        }
        let pattern = CircFixedStructPattern::from_circ(pattern_circ.clone(), node_equality());
        let pmatcher = PatternMatcher::new(pattern, target_circ.dag_ref());

        group.bench_function(BenchmarkId::new("Sequential", i), |b| {
            b.iter(|| {
                let ms = pmatcher.find_matches().collect::<Vec<_>>();
                assert_eq!(ms.len(), i)
            })
        });
        group.bench_function(BenchmarkId::new("Paralllel", i), |b| {
            b.iter(|| {
                let ms = pmatcher.find_par_matches().collect::<Vec<_>>();
                assert_eq!(ms.len(), i)
            })
        });
    }
    group.finish();
}

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
//         .append_op(Op::H, &vec![PortIndex::new(0)])
//         .unwrap();
//     pattern_circ
//         .append_op(Op::H, &vec![PortIndex::new(1)])
//         .unwrap();
//     pattern_circ
//         .append_op(Op::CX, &vec![PortIndex::new(0), PortIndex::new(1)])
//         .unwrap();
//     pattern_circ
//         .append_op(Op::H, &vec![PortIndex::new(0)])
//         .unwrap();
//     pattern_circ
//         .append_op(Op::H, &vec![PortIndex::new(1)])
//         .unwrap();
//     let pattern_boundary = pattern_circ.boundary();

//     let mut target_circ = Circuit::with_uids(qubits);
//     target_circ
//         .append_op(Op::H, &vec![PortIndex::new(0)])
//         .unwrap();
//     target_circ
//         .append_op(Op::H, &vec![PortIndex::new(1)])
//         .unwrap();
//     target_circ
//         .append_op(Op::CX, &vec![PortIndex::new(0), PortIndex::new(1)])
//         .unwrap();
//     target_circ
//         .append_op(Op::H, &vec![PortIndex::new(0)])
//         .unwrap();
//     target_circ
//         .append_op(Op::H, &vec![PortIndex::new(1)])
//         .unwrap();
//     let mut group = c.benchmark_group("PatternMatch");
//     for i in (100..1000).step_by(100) {
//         for _ in 0..100 {
//             target_circ
//                 .append_op(Op::CX, &vec![PortIndex::new(0), PortIndex::new(1)])
//                 .unwrap();
//             target_circ
//                 .append_op(Op::H, &vec![PortIndex::new(0)])
//                 .unwrap();
//             target_circ
//                 .append_op(Op::H, &vec![PortIndex::new(1)])
//                 .unwrap();

//             pattern_circ
//                 .append_op(Op::CX, &vec![PortIndex::new(0), PortIndex::new(1)])
//                 .unwrap();
//             pattern_circ
//                 .append_op(Op::H, &vec![PortIndex::new(0)])
//                 .unwrap();
//             pattern_circ
//                 .append_op(Op::H, &vec![PortIndex::new(1)])
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

criterion_group!(benches, pattern_match_bench_par);
criterion_main!(benches);
