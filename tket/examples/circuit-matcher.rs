//! Example of a circuit matcher to match Clifford circuits.

use std::{hash::Hash, ops::Range};

use hugr::HugrView;
use itertools::Itertools;
use tket::{
    rewrite::matcher::{CircuitMatcher, MatchContext, MatchOutcome, MatchingOptions, OpArg},
    serialize::TKETDecode,
    TketOp,
};
use tket_json_rs::SerialCircuit;

/// A matcher finding Clifford subcircuits with number of CX gates in the given
/// range.
struct CliffordMatcher {
    allowed_num_cx: Range<usize>,
}

/// Data that we track during matching.
#[derive(Debug, Clone, Default, Hash)]
struct MatchInfo {
    /// Number of two-qubit Clifford gates matched so far.
    num_cx: usize,
    /// Trace of the matching process, for debugging.
    debug_info: String,
}

impl CircuitMatcher for CliffordMatcher {
    type PartialMatchInfo = MatchInfo;
    type MatchInfo = MatchInfo;

    fn match_tket_op(
        &self,
        op: TketOp,
        op_args: &[OpArg],
        match_context: MatchContext<Self::PartialMatchInfo, impl HugrView>,
    ) -> MatchOutcome<Self::PartialMatchInfo, Self::MatchInfo> {
        let MatchInfo {
            mut num_cx,
            mut debug_info,
        } = match_context.match_info;

        let mut trace_debug_info = |s: &str| {
            debug_info.push_str(s);
        };

        trace_debug_info(&format!("Got op: {op:?}\n"));
        trace_debug_info(&format!("Got args: {op_args:?}\n"));

        use TketOp::*;
        match op {
            CX | CY | CZ => {
                // at every CX, return a complete match
                num_cx += 1;
                if num_cx >= self.allowed_num_cx.end {
                    MatchOutcome::stop() // no larger match possible
                } else {
                    trace_debug_info("=> Matched two-qb Clifford\n");

                    let match_outcome = MatchOutcome::default().proceed(MatchInfo {
                        num_cx,
                        debug_info: debug_info.clone(),
                    });

                    if self.allowed_num_cx.contains(&num_cx) {
                        match_outcome.complete(MatchInfo { num_cx, debug_info })
                    } else {
                        match_outcome
                    }
                }
            }
            H | S | Sdg | X | Y | Z | V | Vdg => {
                // Keep extending the match
                trace_debug_info("=> Matched single-qb Clifford\n");
                MatchOutcome::default().proceed(MatchInfo { num_cx, debug_info })
            }
            Rx | Ry | Rz => {
                // If clifford, keep extending
                let OpArg::ConstF64(half_turns) = op_args[1] else {
                    panic!("expected angle rotation argument");
                };
                const EPS: f64 = 1e-6;
                let quarter_turns = 2. * half_turns + EPS / 2.;
                if quarter_turns.fract() < EPS {
                    trace_debug_info("=> Matched Clifford-angle rotation\n");
                    MatchOutcome::default().proceed(MatchInfo { num_cx, debug_info })
                } else {
                    trace_debug_info("=> Skipped non-Clifford rotation\n");
                    MatchOutcome::default().skip(MatchInfo { num_cx, debug_info })
                }
            }
            _ => {
                trace_debug_info("=> Skipped unknown gate\n");
                MatchOutcome::default().skip(MatchInfo { num_cx, debug_info })
            }
        }
    }
}

fn main() {
    const CIRCUIT: &str = r#"{"bits": [], "commands": [{"args": [["q", [0]], ["q", [1]]], "op": {"type": "CX"}}, {"args": [["q", [0]]], "op": {"params": ["0.5"], "type": "Rz"}}, {"args": [["q", [1]]], "op": {"type": "V"}}, {"args": [["q", [0]], ["q", [2]]], "op": {"type": "CX"}}, {"args": [["q", [0]], ["q", [1]]], "op": {"type": "CX"}}, {"args": [["q", [2]]], "op": {"type": "S"}}, {"args": [["q", [0]]], "op": {"params": ["0.111"], "type": "Rz"}}, {"args": [["q", [2]]], "op": {"type": "T"}}, {"args": [["q", [1]], ["q", [2]]], "op": {"type": "CX"}}, {"args": [["q", [1]]], "op": {"type": "T"}}, {"args": [["q", [2]]], "op": {"type": "S"}}, {"args": [["q", [0]], ["q", [1]]], "op": {"type": "CX"}}], "created_qubits": [], "discarded_qubits": [], "implicit_permutation": [[["q", [0]], ["q", [0]]], [["q", [1]], ["q", [1]]], [["q", [2]], ["q", [2]]]], "phase": "0.0", "qubits": [["q", [0]], ["q", [1]], ["q", [2]]]}"#;
    let ser_circ: SerialCircuit = serde_json::from_str(CIRCUIT).unwrap();
    let circuit = ser_circ.decode().unwrap();

    let matcher = CliffordMatcher {
        allowed_num_cx: 2..4,
    };

    let matches = matcher.as_hugr_matcher().get_all_matches(
        &circuit,
        &MatchingOptions::with_deduplication().only_maximal_matches(),
    );

    println!("Found {} matches", matches.len());

    for (i, (subgraph, info)) in matches.into_iter().enumerate() {
        println!("==== Match {} ====\n{}", i, info.debug_info);
        println!("  Num CX: {}", info.num_cx);
        println!(
            "  Matched nodes: {:?}\n",
            subgraph.nodes().iter().sorted().collect_vec()
        );
    }
}
