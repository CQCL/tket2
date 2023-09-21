use std::{collections::HashSet, fs, time::Instant};

use hugr::{Hugr, HugrView};
use lazy_static::lazy_static;
use tket2::{
    circuit::{Circuit, OpType},
    json::load_tk1_json_file,
    optimiser::TasoOptimiser,
    rewrite::{strategy::ExhaustiveRewriteStrategy, ECCRewriter},
    T2Op,
};

#[derive(Debug, serde::Deserialize)]
struct QuartzGateCount {
    circ_name: String,
    count_before: usize,
    count_after: usize,
}

#[derive(Debug, serde::Serialize)]
struct ElapsedTime {
    circ_name: String,
    elapsed_s: f64,
    opt_count: usize,
}

lazy_static! {
    static ref QUANTUM_GATES: HashSet<T2Op> = FromIterator::from_iter([
        T2Op::H,
        T2Op::CX,
        T2Op::T,
        T2Op::S,
        T2Op::X,
        T2Op::Y,
        T2Op::Z,
        T2Op::Tdg,
        T2Op::Sdg,
        T2Op::RzF64,
        T2Op::TK1,
    ]);
}

fn num_cx_gates(circ: &Hugr) -> usize {
    circ.commands()
        .filter(|cmd| {
            let n = cmd.node();
            let OpType::LeafOp(op) = circ.get_optype(n) else {
                return false;
            };
            let op: T2Op = op.clone().try_into().unwrap();
            op == T2Op::CX
        })
        .count()
}

fn main() {
    let gate_counts_f = fs::File::open("gate_counts_quartz.csv").unwrap();
    let mut gate_counts_csv = csv::Reader::from_reader(gate_counts_f);

    let timings_f = fs::File::create("results.csv").unwrap();
    let mut timings_csv = csv::Writer::from_writer(timings_f);

    let rewriter = load_rewriter();
    let strategy = ExhaustiveRewriteStrategy::default();
    let taso = TasoOptimiser::new(rewriter, strategy, num_cx_gates);

    for result in gate_counts_csv.deserialize() {
        let record: QuartzGateCount = result.unwrap();
        let circ = load_tk1_json_file(format!("circuits/{}.json", record.circ_name)).unwrap();
        println!("Optimising {}", record.circ_name);
        let start_time = Instant::now();
        let opt_circ = taso
            .optimise_with_default_log(&circ, Some(15), 4.try_into().unwrap())
            .unwrap();
        timings_csv
            .serialize(ElapsedTime {
                circ_name: record.circ_name.clone(),
                elapsed_s: start_time.elapsed().as_secs_f64(),
                opt_count: num_cx_gates(&opt_circ),
            })
            .unwrap();
    }
    println!("Done!");
}

fn load_rewriter() -> ECCRewriter {
    println!("Compiling rewriter...");
    ECCRewriter::try_from_eccs_json_file("Nam_6_3_complete_ECC_set.json").unwrap()
}
