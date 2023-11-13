use tket2::extension::REGISTRY;
use tket2::json::load_tk1_json_reader;
use tket2::optimiser::DefaultBadgerOptimiser;

/// This hugr corresponds to the qasm circuit:
///
/// ```skip
/// OPENQASM 2.0;
/// include "qelib1.inc";
///
/// qreg q[4];
/// cx q[0],q[1];
/// cx q[2],q[3];
/// h q[1];
/// h q[3];
/// cx q[2],q[1];
/// cx q[0],q[3];
/// h q[0];
/// h q[1];
/// h q[2];
/// h q[3];
/// ```
const TK1: &str = r#"{"phase":"0.0","commands":[{"op":{"type":"CX","n_qb":2,"signature":["Q","Q"]},"args":[["q",[2]],["q",[3]]]},{"op":{"type":"H","n_qb":1,"signature":["Q"]},"args":[["q",[3]]]},{"op":{"type":"CX","n_qb":2,"signature":["Q","Q"]},"args":[["q",[0]],["q",[1]]]},{"op":{"type":"H","n_qb":1,"signature":["Q"]},"args":[["q",[1]]]},{"op":{"type":"CX","n_qb":2,"signature":["Q","Q"]},"args":[["q",[2]],["q",[1]]]},{"op":{"type":"H","n_qb":1,"signature":["Q"]},"args":[["q",[1]]]},{"op":{"type":"H","n_qb":1,"signature":["Q"]},"args":[["q",[2]]]},{"op":{"type":"CX","n_qb":2,"signature":["Q","Q"]},"args":[["q",[0]],["q",[3]]]},{"op":{"type":"H","n_qb":1,"signature":["Q"]},"args":[["q",[3]]]},{"op":{"type":"H","n_qb":1,"signature":["Q"]},"args":[["q",[0]]]}],"qubits":[["q",[0]],["q",[1]],["q",[2]],["q",[3]]],"bits":[],"implicit_permutation":[[["q",[0]],["q",[0]]],[["q",[1]],["q",[1]]],[["q",[2]],["q",[2]]],[["q",[3]],["q",[3]]]]}"#;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    //let circ: Hugr = serde_json::from_str(HUGR)?;
    let circ = load_tk1_json_reader(TK1.as_bytes())?;

    // This fails.
    let badger = DefaultBadgerOptimiser::default_with_rewriter_binary("test_files/nam_6_3.rwr")?;

    let mut new_circ = badger.optimise(&circ, Some(0), 1.try_into().unwrap(), false, 10);

    new_circ.update_validate(&REGISTRY)?;

    Ok(())
}
