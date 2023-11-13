use hugr::hugr::views::SiblingSubgraph;
use hugr::hugr::Rewrite;
use itertools::Itertools;
use tket2::extension::REGISTRY;
use tket2::json::load_tk1_json_reader;
use tket2::Circuit;

const CIRCUIT: &str = r#"
{
    "phase":"0.0",
    "commands": [
        {"op":{"type":"CX","n_qb":2,"signature":["Q","Q"]},"args":[["q",[2]],["q",[0]]]},
        {"op":{"type":"CX","n_qb":2,"signature":["Q","Q"]},"args":[["q",[1]],["q",[0]]]},
        {"op":{"type":"U3","params":["0.5","0","0.5"],"signature":["Q"]},"args":[["q",[2]]]}
    ],
    "qubits": [["q",[0]],["q",[1]],["q",[2]]],
    "bits":[],
    "implicit_permutation":[[["q",[0]],["q",[0]]],[["q",[1]],["q",[1]]],[["q",[2]],["q",[2]]]]
}
"#;

const REPLACEMENT: &str = r#"
{
    "phase":"0.0",
    "commands": [
        {"op":{"type":"CX","n_qb":2,"signature":["Q","Q"]},"args":[["q",[1]],["q",[0]]]},
        {"op":{"type":"CX","n_qb":2,"signature":["Q","Q"]},"args":[["q",[2]],["q",[0]]]}
    ],
    "qubits": [["q",[0]],["q",[1]],["q",[2]]],
    "bits":[],
    "implicit_permutation":[[["q",[0]],["q",[0]]],[["q",[1]],["q",[1]]],[["q",[2]],["q",[2]]]]
}
"#;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let circ = load_tk1_json_reader(CIRCUIT.as_bytes())?;
    let replacement = load_tk1_json_reader(REPLACEMENT.as_bytes())?;

    let (cx0, cx1) = circ
        .commands()
        .take(2)
        .map(|c| c.node())
        .collect_tuple()
        .unwrap();

    let subgraph = SiblingSubgraph::try_from_nodes([cx0, cx1], &circ)?;
    let rewrite = subgraph.create_simple_replacement(&circ, replacement)?;

    let mut new_circ = circ.clone();
    rewrite.apply(&mut new_circ).unwrap();

    // This fails.
    new_circ.update_validate(&REGISTRY)?;

    Ok(())
}
