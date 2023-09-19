use std::{collections::HashSet, fs, time::Instant};

use hugr::{Hugr, HugrView};
use lazy_static::lazy_static;
use tket2::{
    circuit::{Circuit, OpType},
    json::load_tk1_json_file,
    passes::taso::taso,
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
    timed_out: bool,
    quartz_count: usize,
    opt_count: usize,
    panicked: bool,
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

fn num_q_gates(circ: &Hugr) -> usize {
    circ.commands()
        .filter(|cmd| {
            let n = cmd.node();
            let OpType::LeafOp(op) = circ.get_optype(n) else {
                return false;
            };
            let op = op.clone().try_into().unwrap();
            QUANTUM_GATES.contains(&op)
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

    for result in gate_counts_csv.deserialize() {
        let record: QuartzGateCount = result.unwrap();
        let circ = load_tk1_json_file(format!("circuits/{}.json", record.circ_name)).unwrap();
        if num_q_gates(&circ) != record.count_before {
            panic!(
                "Mismatched gate count for {}: expected {}, got {}",
                record.circ_name,
                record.count_before,
                num_q_gates(&circ)
            );
        }
        println!("Optimising {}", record.circ_name);
        let start_time = Instant::now();
        match taso(
            circ,
            &rewriter,
            &strategy,
            num_q_gates,
            Some(1000),
            &record.circ_name,
            record.count_after,
        ) {
            Ok((opt_circ, timed_out)) => timings_csv
                .serialize(ElapsedTime {
                    circ_name: record.circ_name.clone(),
                    elapsed_s: start_time.elapsed().as_secs_f64(),
                    timed_out,
                    quartz_count: record.count_after,
                    opt_count: num_q_gates(&opt_circ),
                    panicked: false,
                })
                .unwrap(),
            Err(()) => timings_csv
                .serialize(ElapsedTime {
                    circ_name: record.circ_name.clone(),
                    elapsed_s: start_time.elapsed().as_secs_f64(),
                    timed_out: false,
                    quartz_count: record.count_after,
                    opt_count: 0,
                    panicked: false,
                })
                .unwrap(),
        };
    }
    println!("Done!");
}

fn load_rewriter() -> ECCRewriter {
    println!("Compiling rewriter...");
    ECCRewriter::from_eccs_json_file("Nam_6_3_complete_ECC_set.json")
}
