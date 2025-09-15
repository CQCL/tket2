//! Using Badger to perform Hadamard cancellation.

use hugr::{
    builder::{endo_sig, DFGBuilder, Dataflow, DataflowHugr},
    extension::prelude::qb_t,
    HugrView,
};
use tket::{
    op_matches,
    optimiser::BadgerOptimiser,
    resource::{CircuitUnit, ResourceScope},
    rewrite::{
        matcher::{CircuitMatcher, MatchContext, MatchOutcome},
        replacer::CircuitReplacer,
        strategy::LexicographicCostFunction,
        MatchReplaceRewriter,
    },
    Circuit, Subcircuit, TketOp,
};

/// A matcher that matches two Hadamard gates in a row.
#[derive(Clone, Copy, Debug)]
struct TwoHMatcher;

/// A replacement that replaces two Hadamard gates in a row with the identity.
#[derive(Clone, Copy, Debug)]
struct HadamardCancellation;

/// State to keep track of how much has been matched so far.
#[derive(Clone, Copy, Default, PartialEq, Eq, Hash)]
enum PartialMatchState {
    /// No hadamard matched so far.
    #[default]
    NoMatch,
    /// One hadamard matched so far.
    MatchedOne,
}

impl CircuitMatcher for TwoHMatcher {
    type PartialMatchInfo = PartialMatchState;
    type MatchInfo = ();

    fn match_tket_op<H: HugrView>(
        &self,
        op: tket::TketOp,
        _op_args: &[CircuitUnit<H::Node>],
        match_context: MatchContext<Self::PartialMatchInfo, H>,
    ) -> MatchOutcome<Self::PartialMatchInfo, Self::MatchInfo> {
        if op != TketOp::H {
            // We are not intersted in matching this op
            return MatchOutcome::stop();
        }

        match match_context.match_info {
            PartialMatchState::NoMatch => {
                // Making progress! Proceed to matching the second Hadamard
                MatchOutcome::default().proceed(PartialMatchState::MatchedOne)
            }
            PartialMatchState::MatchedOne => {
                // We have matched two Hadamards, so we can report the match
                MatchOutcome::default()
                    .complete(()) // report a full match
                    .skip(PartialMatchState::MatchedOne) // consider skipping
                                                         // this op (and match
                                                         // another one)
            }
        }
    }
}

impl CircuitReplacer<()> for HadamardCancellation {
    fn replace_match<H: hugr::HugrView>(
        &self,
        subcircuit: &Subcircuit<H::Node>,
        circuit: &ResourceScope<H>,
        _match_info: (),
    ) -> Vec<tket::Circuit> {
        let hugr = circuit.hugr();
        // subgraph should be a pair of Hadamards
        assert_eq!(subcircuit.node_count(circuit), 2);
        assert!(subcircuit
            .nodes(circuit)
            .all(|n| op_matches(hugr.get_optype(n), TketOp::H)));

        // The right hand side of the rewrite is just an empty one-qubit circuit
        let h = DFGBuilder::new(endo_sig(qb_t())).unwrap();
        let inps = h.input_wires();
        let empty_circ = h.finish_hugr_with_outputs(inps).unwrap();

        vec![Circuit::new(empty_circ)]
    }
}

/// A 4-qubit circuit made of three layers
///  - layer of 4x Hadamards on each qubit,
///  - layer of CX gates between pairs of qubits,
///  - layer of 4x Hadamards on each qubit,
fn h_cx_h() -> Circuit {
    let mut h = DFGBuilder::new(endo_sig(vec![qb_t(); 4])).unwrap();
    let qbs = h.input_wires();
    let mut circ = h.as_circuit(qbs);

    for _ in 0..4 {
        for i in 0..4 {
            circ.append(TketOp::H, [i]).unwrap();
        }
    }

    for i in (0..4).step_by(2) {
        circ.append(TketOp::CX, [i, i + 1]).unwrap();
    }

    for _ in 0..4 {
        for i in 0..4 {
            circ.append(TketOp::H, [i]).unwrap();
        }
    }

    let qbs = circ.finish();
    Circuit::new(h.finish_hugr_with_outputs(qbs).unwrap())
}

fn main() {
    let rewriter = MatchReplaceRewriter::new(TwoHMatcher, HadamardCancellation, None);

    let optimiser =
        BadgerOptimiser::new(rewriter, LexicographicCostFunction::default_cx_strategy());

    let circuit = h_cx_h();

    let optimised = optimiser.optimise(&circuit, Default::default());

    // Only CX gates are left
    assert_eq!(optimised.num_operations(), 2);
    assert!(optimised
        .operations()
        .all(|cmd| op_matches(cmd.optype(), TketOp::CX)));

    println!("Success!");
}
