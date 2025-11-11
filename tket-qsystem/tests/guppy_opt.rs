//! Tests optimizing Guppy-generated programs.

use std::fs;
use std::io::BufReader;
use std::path::Path;
use tket::extension::{TKET1_EXTENSION_ID, TKET_EXTENSION_ID};

use hugr::algorithms::ComposablePass;
use hugr::{Hugr, HugrView};
use rstest::{fixture, rstest};
use tket::passes::NormalizeGuppy;
use tket_qsystem::QSystemPass;

const GUPPY_EXAMPLES_DIR: &str = "../test_files/guppy_optimization";

fn load_guppy_circuit(name: &str) -> Hugr {
    let file = Path::new(GUPPY_EXAMPLES_DIR).join(format!("{name}/{name}.hugr"));
    let reader = fs::File::open(file).unwrap();
    let reader = BufReader::new(reader);
    Hugr::load(reader, None).unwrap()
}

#[fixture]
fn guppy_angles() -> Hugr {
    load_guppy_circuit("angles")
}

#[fixture]
fn guppy_false_branch() -> Hugr {
    load_guppy_circuit("false_branch")
}

#[fixture]
fn guppy_nested() -> Hugr {
    load_guppy_circuit("nested")
}

#[fixture]
fn guppy_ranges() -> Hugr {
    load_guppy_circuit("ranges")
}

#[fixture]
fn guppy_simple_cx() -> Hugr {
    load_guppy_circuit("simple_cx")
}

fn count_gates(h: &impl HugrView) -> (usize, usize) {
    let mut non_alloc = 0;
    let mut alloc = 0;
    for n in h.nodes() {
        if let Some(eop) = h.get_optype(n).as_extension_op() {
            if [TKET_EXTENSION_ID, TKET1_EXTENSION_ID].contains(eop.extension_id()) {
                if ["tket.quantum.QFree", "tket.quantum.QAlloc"]
                    .contains(&eop.qualified_id().as_str())
                {
                    alloc += 1;
                } else {
                    non_alloc += 1;
                }
            }
        }
    }
    (non_alloc, alloc)
}

/// Run some simple optimization passes on the guppy-generated HUGRs and validate the result.
///
/// This test is intended to check the current status of the Guppy optimization passes.
///
#[rstest]
#[case::angles(guppy_angles(), (5, 1))]
#[case::false_branch(guppy_false_branch(), (3, 1))]
#[case::nested(guppy_nested(), (6, 3))]
#[case::ranges(guppy_ranges(), (8, 4))]
#[case::simple_cx(guppy_simple_cx(), (4, 2))]
#[cfg_attr(miri, ignore)] // Opening files is not supported in (isolated) miri
fn optimise_guppy(#[case] mut hugr: Hugr, #[case] before: (usize, usize)) {
    NormalizeGuppy::default().run(&mut hugr).unwrap();
    assert_eq!(count_gates(&hugr), before);

    // TODO: Run pytket passes here, and check that the circuit is as optimized as possible at this point.
    //
    // Most example circuits optimize to identity functions, so it may be possible to check for that.

    // Lower to QSystem. This may blow up the HUGR size.
    QSystemPass::default().run(&mut hugr).unwrap();

    hugr.validate().unwrap_or_else(|e| panic!("{e}"));
}
