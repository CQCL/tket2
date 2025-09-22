//! Tests optimizing Guppy-generated programs.

use std::convert::Infallible;
use std::fs;
use std::io::BufReader;
use std::path::Path;

use hugr::algorithms::const_fold::ConstFoldError;
use hugr::algorithms::inline_dfgs::InlineDFGsPass;
use hugr::algorithms::merge_bbs::merge_basic_blocks;
use hugr::algorithms::untuple::{UntupleError, UntupleRecursive};
use hugr::algorithms::ComposablePass;
use hugr::algorithms::UntuplePass;
use hugr::algorithms::{RemoveDeadFuncsError, RemoveDeadFuncsPass};
use hugr::hugr::hugrmut::HugrMut;
use hugr::hugr::patch::inline_dfg::InlineDFGError;
use hugr::hugr::views::RootCheckable;
use hugr::{Hugr, HugrView};
use itertools::Itertools;
use rstest::{fixture, rstest};
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
    // Merge basic blocks with deterministic branching.
    //
    // This should be a composable pass in hugr-passes.
    let cfgs = hugr
        .entry_descendants()
        .filter(|&n| hugr.get_optype(n).is_cfg())
        .collect_vec();
    for cfg in cfgs {
        let mut rerooted = hugr.with_entrypoint_mut(cfg);
        let cfg = (&mut rerooted).try_into_checked().unwrap();
        merge_basic_blocks(cfg);
    }

    let untuple = UntuplePass::new(UntupleRecursive::Recursive);
    // TODO: Constant folding fails on some test cases (it leaves qubit ports disconnected).
    //let const_fold = ConstantFoldPass::default();
    //let dce = DeadCodeElimPass::<Hugr>::default();
    let dead_funcs = RemoveDeadFuncsPass::default();
    let inline = InlineDFGsPass;
    // TODO: We are missing an InlineCFGs pass here.

    /// Helper error accumulator
    #[derive(derive_more::Error, Debug, derive_more::Display, derive_more::From)]
    enum OptErrors {
        ConstantFold(ConstFoldError),
        Untuple(UntupleError),
        DeadFuncs(RemoveDeadFuncsError),
        Dce(Infallible),
        Inline(InlineDFGError),
    }

    untuple
        .map_err(OptErrors::Untuple)
        //.then::<_, OptErrors>(const_fold)
        //.then::<_, OptErrors>(dce)
        .then::<_, OptErrors>(dead_funcs)
        .then::<_, OptErrors>(inline)
        .run(&mut hugr)
        .unwrap();

    // TODO: Run pytket passes here, and check that the circuit is as optimized as possible at this point.
    //
    // Most example circuits optimize to identity functions, so it may be possible to check for that.

    // Lower to QSystem. This may blow up the HUGR size.
    QSystemPass::default()
        //.with_constant_fold(true)
        .run(&mut hugr)
        .unwrap();

    hugr.validate().unwrap_or_else(|e| panic!("{e}"));
}
