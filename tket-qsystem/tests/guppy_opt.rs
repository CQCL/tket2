//! Tests optimizing Guppy-generated programs.

use rayon::iter::ParallelIterator;
use smol_str::SmolStr;
use std::collections::HashMap;
use std::fs;
use std::io::BufReader;
use std::path::Path;
use tket::extension::{TKET1_EXTENSION_ID, TKET_EXTENSION_ID};

use hugr::algorithms::ComposablePass;
use hugr::{Hugr, HugrView};
use rstest::{fixture, rstest};
use tket::passes::NormalizeGuppy;
use tket::serialize::pytket::{EncodeOptions, EncodedCircuit};
use tket::Circuit;

use tket1_passes::Tket1Circuit;
use tket_qsystem::QSystemPass;

const GUPPY_EXAMPLES_DIR: &str = "../test_files/guppy_optimization";

fn load_guppy_circuit(path: &str) -> std::io::Result<Hugr> {
    let file = Path::new(GUPPY_EXAMPLES_DIR).join(format!("{path}.hugr"));
    let reader = fs::File::open(file)?;
    let reader = BufReader::new(reader);
    Ok(Hugr::load(reader, None).unwrap())
}

#[fixture]
fn guppy_angles() -> Hugr {
    load_guppy_circuit("angles/angles").unwrap()
}

#[fixture]
fn guppy_false_branch() -> Hugr {
    load_guppy_circuit("false_branch/false_branch").unwrap()
}

#[fixture]
fn guppy_nested() -> Hugr {
    load_guppy_circuit("nested/nested").unwrap()
}

#[fixture]
fn guppy_ranges() -> Hugr {
    load_guppy_circuit("ranges/ranges").unwrap()
}

#[fixture]
fn guppy_simple_cx() -> Hugr {
    load_guppy_circuit("simple_cx/simple_cx").unwrap()
}

fn run_pytket(h: &mut Hugr) {
    let circ = Circuit::new(h);
    let mut encoded =
        EncodedCircuit::new(&circ, EncodeOptions::new().with_subcircuits(true)).unwrap();

    encoded
        .par_iter_mut()
        .for_each(|(_region, serial_circuit)| {
            let mut circuit_ptr = Tket1Circuit::from_serial_circuit(serial_circuit).unwrap();
            circuit_ptr
                .clifford_simp(tket_json_rs::OpType::CX, true)
                .unwrap();
            *serial_circuit = circuit_ptr.to_serial_circuit().unwrap();
        });

    encoded.reassemble_inplace(circ.into_hugr(), None).unwrap();
}

fn count_gates(h: &impl HugrView) -> HashMap<SmolStr, usize> {
    let mut counts = HashMap::new();
    for n in h.nodes() {
        if let Some(eop) = h.get_optype(n).as_extension_op() {
            if [TKET_EXTENSION_ID, TKET1_EXTENSION_ID].contains(eop.extension_id()) {
                *counts.entry(eop.qualified_id()).or_default() += 1;
            }
        }
    }
    counts
}

/// Run some simple optimization passes on the guppy-generated HUGRs and validate the result.
///
/// This test is intended to check the current status of the Guppy optimization passes.
///

#[rstest]
#[case::false_branch(guppy_false_branch(), [
    ("tket.quantum.H", 2), ("tket.quantum.QAlloc", 1), ("tket.quantum.MeasureFree", 1)
], [
    ("TKET1.tk1op", 1), ("tket.quantum.QAlloc", 1), ("tket.quantum.H", 1), ("tket.quantum.MeasureFree", 1)
])]
#[case::simple_cx(guppy_simple_cx(), [
    ("tket.quantum.QAlloc", 2), ("tket.quantum.CX", 2), ("tket.quantum.MeasureFree", 2)
], [
    ("tket.quantum.MeasureFree", 2), ("tket.quantum.QAlloc", 2)
])]
#[cfg_attr(miri, ignore)] // Opening files is not supported in (isolated) miri
fn optimise_guppy(
    #[case] mut hugr: Hugr,
    #[case] before: impl IntoIterator<Item = (impl Into<SmolStr>, usize)>,
    #[case] after: impl IntoIterator<Item = (impl Into<SmolStr>, usize)>,
) {
    NormalizeGuppy::default().run(&mut hugr).unwrap();
    let before = before.into_iter().map(|(k, v)| (k.into(), v)).collect();
    assert_eq!(count_gates(&hugr), before);

    run_pytket(&mut hugr);

    let after = after.into_iter().map(|(k, v)| (k.into(), v)).collect();
    assert_eq!(count_gates(&hugr), after);

    // Lower to QSystem. This may blow up the HUGR size.
    QSystemPass::default().run(&mut hugr).unwrap();

    hugr.validate().unwrap_or_else(|e| panic!("{e}"));
}

/// Checks that pytket does nothing for these examples.
/// We include gate counts for after the NormalizeGuppy step
/// as our flattening is not sufficient to match the .flat.hugr
#[rstest]
#[case::angles(guppy_angles(), [
    ("tket.quantum.H", 2), ("tket.quantum.QAlloc", 1), ("tket.quantum.Rz", 2), ("tket.quantum.MeasureFree", 1)
])]
#[case::nested(guppy_nested(), [
    ("tket.quantum.CZ", 1), ("tket.quantum.H", 2), ("tket.quantum.QAlloc", 3), ("tket.quantum.MeasureFree", 3)
])]
#[case::ranges(guppy_ranges(), [
    ("tket.quantum.QAlloc", 4), ("tket.quantum.MeasureFree", 4), ("tket.quantum.H", 2), ("tket.quantum.CX", 2)
])]
#[cfg_attr(miri, ignore)] // Opening files is not supported in (isolated) miri
fn no_optimise_guppy<'a>(
    #[case] hugr: Hugr,
    #[case] before_after: impl IntoIterator<Item = (&'a str, usize)> + Clone,
) {
    optimise_guppy(hugr, before_after.clone(), before_after);
}

/// Check that each example optimizes to the full extent given by the .opt (and .flat) .hugr files.
#[rstest]
#[case::angles("angles")]
#[case::false_branch("false_branch")]
#[case::simple_cx("simple_cx")]
#[case::nested("nested")]
#[case::ranges("ranges")]
#[should_panic] // This does not yet pass for any case!
fn optimise_guppy_full(#[case] name: &str) {
    let hugr = load_guppy_circuit(&format!("{name}/{name}")).unwrap();
    let flat = load_guppy_circuit(&format!("{name}/{name}.flat")).unwrap_or(hugr.clone());
    let opt = load_guppy_circuit(&format!("{name}/{name}.opt")).unwrap();

    optimise_guppy(hugr, count_gates(&flat), count_gates(&opt))
}
