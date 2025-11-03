//! Tests optimizing Guppy-generated programs.

use std::fs;
use std::io::BufReader;
use std::path::Path;

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

/// Run some simple optimization passes on the guppy-generated HUGRs and validate the result.
///
/// This test is intended to check the current status of the Guppy optimization passes.
///
#[rstest]
#[case::angles(guppy_angles())]
#[case::false_branch(guppy_false_branch())]
#[case::nested(guppy_nested())]
#[case::ranges(guppy_ranges())]
#[case::simple_cx(guppy_simple_cx())]
#[cfg_attr(miri, ignore)] // Opening files is not supported in (isolated) miri
fn optimise_guppy(#[case] mut hugr: Hugr) {
    NormalizeGuppy::default().run(&mut hugr).unwrap();

    // TODO: Run pytket passes here, and check that the circuit is as optimized as possible at this point.
    //
    // Most example circuits optimize to identity functions, so it may be possible to check for that.

    // Lower to QSystem. This may blow up the HUGR size.
    QSystemPass::default().run(&mut hugr).unwrap();

    hugr.validate().unwrap_or_else(|e| panic!("{e}"));
}
