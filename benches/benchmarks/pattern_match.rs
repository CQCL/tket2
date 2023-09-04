use std::path::{Path, PathBuf};

use criterion::measurement::Measurement;
use criterion::{black_box, criterion_group, BenchmarkGroup, BenchmarkId, Criterion, SamplingMode};
use hugr::hugr::views::DescendantsGraph;
use hugr::ops::handle::DfgID;
use hugr::Hugr;
use hugr::HugrView;
use itertools::Itertools;
use tket2::circuit::{HierarchyView};
use tket2::json::load_tk1_json_file;
use tket2::portmatching::{CircuitMatcher, CircuitPattern};

static PATTERNS_FOLDER: &str = "test_files/T_Tdg_H_X_CX_complete_ECC_set";
static CIRCUIT_FILE: &str = "test_files/barenco_tof_5";

fn to_circ(h: &Hugr) -> DescendantsGraph<'_, DfgID> {
    DescendantsGraph::new(h, h.root())
}

fn path_as_int(path: &Path) -> usize {
    let stem = path.file_stem().expect("invalid path");
    stem.to_str()
        .expect("not a valid path name")
        .parse()
        .expect("file name not a number")
}

fn bench_pattern_match(c: &mut Criterion) {
    let mut group = c.benchmark_group("Pattern matching");
    group.sampling_mode(SamplingMode::Flat);
    group.sample_size(10);
    pattern_match_tk2_bench(&mut group);
    #[cfg(feature = "quartz-bench")]
    pattern_match_quartz_bench(&mut group);
    group.finish()
}

fn pattern_match_tk2_bench<M: Measurement>(group: &mut BenchmarkGroup<'_, M>) {
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
        .sorted_by_key(|path| path_as_int(path));

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
include!("../quartz/bindings.rs");

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

criterion_group! {
    name = benches;
    config = Criterion::default();
    targets =
        bench_pattern_match,
}
