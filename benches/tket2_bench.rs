use std::{hint::black_box, path::PathBuf};

use criterion::{
    criterion_group, criterion_main, measurement::Measurement, BenchmarkGroup, BenchmarkId,
    Criterion, SamplingMode,
};

use hugr::{hugr::views::DescendantsGraph, ops::handle::DfgID, Hugr, HugrView};
use itertools::Itertools;
use tket2::{
    circuit::HierarchyView,
    json::load_tk1_json_file,
    portmatching::{CircuitMatcher, CircuitPattern},
};

static PATTERNS_FOLDER: &str = "test_files/T_Tdg_H_X_CX_complete_ECC_set";
static CIRCUIT_FILE: &str = "test_files/barenco_tof_5";

fn to_circ(h: &Hugr) -> DescendantsGraph<'_, DfgID> {
    DescendantsGraph::new(h, h.root())
}

fn path_as_int(path: &PathBuf) -> usize {
    let stem = path.file_stem().expect("invalid path");
    stem.to_str()
        .expect("not a valid path name")
        .parse()
        .expect("file name not a number")
}

fn pattern_match_bench(c: &mut Criterion) {
    let mut group = c.benchmark_group("Pattern matching");
    group.sampling_mode(SamplingMode::Flat);
    group.sample_size(10);
    pattern_match_tk2_bench(&mut group);
    #[cfg(feature = "quartz-bench")]
    pattern_match_quartz_bench(&mut group);
    group.finish()
}

fn pattern_match_tk2_bench<'a, M: Measurement>(group: &mut BenchmarkGroup<'a, M>) {
    let folder = PathBuf::from(PATTERNS_FOLDER);
    let circuit_file = format!("{}.json", CIRCUIT_FILE);

    let json_files = folder
        .read_dir()
        .expect("Failed to read test files")
        .map(|entry| {
            let entry = entry.expect("Failed to read directory entry");
            entry.path()
        })
        .filter(|path| path.extension().map_or(false, |ext| ext == "json"))
        .sorted_by_key(path_as_int);

    println!("Loading patterns...");
    let all_hugrs = json_files
        .map(|path| load_tk1_json_file(path.to_str().expect("invalid path")))
        .collect::<Result<Vec<_>, _>>()
        .expect("invalid JSON file");
    let all_circs = all_hugrs.iter().map(to_circ).collect_vec();
    let all_patterns = all_circs
        .iter()
        .map(CircuitPattern::try_from_circuit)
        .collect::<Result<Vec<_>, _>>()
        .expect("invalid pattern");
    println!("Loaded {} patterns", all_patterns.len());

    let target_hugr = load_tk1_json_file(&circuit_file).unwrap();
    let target_circ = to_circ(&target_hugr);
    println!("Loaded circuit");

    for n in (200..=2000).step_by(200) {
        // TODO: store matcher as binary
        println!("Building matcher for n = {n}...");
        let matcher = CircuitMatcher::from_patterns(all_patterns[..n].to_vec());

        group.bench_function(BenchmarkId::new("TKET2", n), |b| {
            b.iter(|| {
                black_box(matcher.find_matches(&target_circ));
            })
        });
    }
}

#[cfg(feature = "quartz-bench")]
#[allow(non_camel_case_types)]
include!("quartz/bindings.rs");

#[cfg(feature = "quartz-bench")]
fn pattern_match_quartz_bench<'a, M: Measurement>(group: &mut BenchmarkGroup<'a, M>) {
    use std::ffi::CString;

    let circuit_file = CString::new(format!("{CIRCUIT_FILE}.qasm")).unwrap();
    let folder_name = CString::new(PATTERNS_FOLDER).unwrap();

    let graph = unsafe { load_graph(circuit_file.as_ptr()) };
    let mut n_ops = 0;
    let ops = unsafe { get_ops(graph, &mut n_ops) };

    let mut n_xfers = 0;
    let xfers = unsafe { load_xfers(folder_name.as_ptr(), &mut n_xfers) };

    for n in (200..=2000).step_by(200) {
        group.bench_function(BenchmarkId::new("Quartz", n), |b| {
            b.iter(|| {
                unsafe { black_box(pattern_match(graph, ops, n_ops, xfers, n)) };
            })
        });
    }

    unsafe {
        free_xfers(xfers, n_xfers);
        free_ops(ops);
        free_graph(graph);
    };
}

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
criterion_group!(benches, /*squash_bench,*/ pattern_match_bench);
criterion_main!(benches);
