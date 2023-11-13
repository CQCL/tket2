use hugr::HugrView;
use tket2::extension::REGISTRY;
use tket2::json::load_tk1_json_reader;
use tket2::optimiser::DefaultBadgerOptimiser;

/// This hugr corresponds to the qasm circuit:
///
/// ```skip
/// OPENQASM 2.0;
/// include "qelib1.inc";
///
/// qreg q[5];
/// cx q[4],q[1];
/// cx q[3],q[4];
/// cx q[1],q[2];
/// cx q[4],q[0];
/// u3(0.5*pi,0.0*pi,0.5*pi) q[1];
/// cx q[0],q[2];
/// cx q[3],q[1];
/// cx q[0],q[2];
/// ```
///
/// Note that this cannot be loaded due to https://github.com/CQCL/hugr/issues/683
const _HUGR: &str = r#"{"version":"v0","nodes":[{"parent":0,"input_extensions":["prelude"],"op":"DFG","signature":{"input":[{"t":"Q"},{"t":"Q"},{"t":"Q"},{"t":"Q"},{"t":"Q"}],"output":[{"t":"Q"},{"t":"Q"},{"t":"Q"},{"t":"Q"},{"t":"Q"}],"extension_reqs":[]}},{"parent":0,"input_extensions":["prelude"],"op":"Input","types":[{"t":"Q"},{"t":"Q"},{"t":"Q"},{"t":"Q"},{"t":"Q"}]},{"parent":0,"input_extensions":["prelude"],"op":"Output","types":[{"t":"Q"},{"t":"Q"},{"t":"Q"},{"t":"Q"},{"t":"Q"}]},{"parent":0,"input_extensions":["prelude"],"op":"LeafOp","lop":"CustomOp","extension":"quantum.tket2","op_name":"CX","description":"TKET 2 quantum op: CX","args":[],"signature":null},{"parent":0,"input_extensions":["prelude"],"op":"LeafOp","lop":"CustomOp","extension":"quantum.tket2","op_name":"CX","description":"TKET 2 quantum op: CX","args":[],"signature":null},{"parent":0,"input_extensions":["prelude"],"op":"LeafOp","lop":"CustomOp","extension":"quantum.tket2","op_name":"CX","description":"TKET 2 quantum op: CX","args":[],"signature":null},{"parent":0,"input_extensions":["prelude"],"op":"LeafOp","lop":"CustomOp","extension":"quantum.tket2","op_name":"CX","description":"TKET 2 quantum op: CX","args":[],"signature":null},{"parent":0,"input_extensions":["prelude"],"op":"Const","value":{"v":"Prim","val":{"pv":"Extension","c":[{"c":"ConstF64","value":0.5}]}},"typ":{"t":"Opaque","extension":"arithmetic.float.types","id":"float64","args":[],"bound":"C"}},{"parent":0,"input_extensions":["prelude"],"op":"LoadConstant","datatype":{"t":"Opaque","extension":"arithmetic.float.types","id":"float64","args":[],"bound":"C"}},{"parent":0,"input_extensions":["prelude"],"op":"Const","value":{"v":"Prim","val":{"pv":"Extension","c":[{"c":"ConstF64","value":0.0}]}},"typ":{"t":"Opaque","extension":"arithmetic.float.types","id":"float64","args":[],"bound":"C"}},{"parent":0,"input_extensions":["prelude"],"op":"LoadConstant","datatype":{"t":"Opaque","extension":"arithmetic.float.types","id":"float64","args":[],"bound":"C"}},{"parent":0,"input_extensions":["prelude"],"op":"Const","value":{"v":"Prim","val":{"pv":"Extension","c":[{"c":"ConstF64","value":0.5}]}},"typ":{"t":"Opaque","extension":"arithmetic.float.types","id":"float64","args":[],"bound":"C"}},{"parent":0,"input_extensions":["prelude"],"op":"LoadConstant","datatype":{"t":"Opaque","extension":"arithmetic.float.types","id":"float64","args":[],"bound":"C"}},{"parent":0,"input_extensions":["prelude"],"op":"LeafOp","lop":"CustomOp","extension":"TKET1","op_name":"TKET1 Json Op","description":"An opaque TKET1 operation.","args":[{"tya":"Opaque","arg":{"typ":{"extension":"TKET1","id":"TKET1 Json Payload","args":[],"bound":"E"},"value":{"op":{"type":"U3","params":["0.5","0.0","0.5"],"signature":["Q"]},"num_qubits":1,"num_bits":0,"param_inputs":[0,1,2],"num_params":3}}}],"signature":null},{"parent":0,"input_extensions":["prelude"],"op":"LeafOp","lop":"CustomOp","extension":"quantum.tket2","op_name":"CX","description":"TKET 2 quantum op: CX","args":[],"signature":null},{"parent":0,"input_extensions":["prelude"],"op":"LeafOp","lop":"CustomOp","extension":"quantum.tket2","op_name":"CX","description":"TKET 2 quantum op: CX","args":[],"signature":null},{"parent":0,"input_extensions":["prelude"],"op":"LeafOp","lop":"CustomOp","extension":"quantum.tket2","op_name":"CX","description":"TKET 2 quantum op: CX","args":[],"signature":null}],"edges":[[[1,0],[6,1]],[[1,1],[3,1]],[[1,2],[4,1]],[[1,3],[5,0]],[[1,4],[3,0]],[[3,0],[5,1]],[[3,1],[4,0]],[[4,0],[13,0]],[[4,1],[14,1]],[[5,0],[15,0]],[[5,1],[6,0]],[[6,0],[2,4]],[[6,1],[14,0]],[[7,null],[8,null]],[[8,0],[13,1]],[[9,null],[10,null]],[[10,0],[13,2]],[[11,null],[12,null]],[[12,0],[13,3]],[[13,0],[15,1]],[[14,0],[16,0]],[[14,1],[16,1]],[[15,0],[2,3]],[[15,1],[2,1]],[[16,0],[2,0]],[[16,1],[2,2]]],"metadata":[{"TKET1_JSON.bit_registers":[],"TKET1_JSON.implicit_permutation":[[["q",[0]],["q",[0]]],[["q",[1]],["q",[1]]],[["q",[2]],["q",[2]]],[["q",[3]],["q",[3]]],[["q",[4]],["q",[4]]]],"TKET1_JSON.phase":"0.0","TKET1_JSON.qubit_registers":[["q",[0]],["q",[1]],["q",[2]],["q",[3]],["q",[4]]],"name":null},null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null]}"#;

/// tket1 equivalent to the above HUGR
const TK1: &str = r#"{"phase":"0.0","commands":[{"op":{"type":"CX","n_qb":2,"signature":["Q","Q"]},"args":[["q",[4]],["q",[1]]]},{"op":{"type":"CX","n_qb":2,"signature":["Q","Q"]},"args":[["q",[1]],["q",[2]]]},{"op":{"type":"U3","params":["0.5","0","0.5"],"signature":["Q"]},"args":[["q",[1]]]},{"op":{"type":"CX","n_qb":2,"signature":["Q","Q"]},"args":[["q",[3]],["q",[4]]]},{"op":{"type":"CX","n_qb":2,"signature":["Q","Q"]},"args":[["q",[4]],["q",[0]]]},{"op":{"type":"CX","n_qb":2,"signature":["Q","Q"]},"args":[["q",[0]],["q",[2]]]},{"op":{"type":"CX","n_qb":2,"signature":["Q","Q"]},"args":[["q",[0]],["q",[2]]]},{"op":{"type":"CX","n_qb":2,"signature":["Q","Q"]},"args":[["q",[3]],["q",[1]]]}],"qubits":[["q",[0]],["q",[1]],["q",[2]],["q",[3]],["q",[4]]],"bits":[],"implicit_permutation":[[["q",[0]],["q",[0]]],[["q",[1]],["q",[1]]],[["q",[2]],["q",[2]]],[["q",[3]],["q",[3]]],[["q",[4]],["q",[4]]]]}"#;

#[allow(unused)]
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
    //let circ: Hugr = serde_json::from_str(HUGR)?;
    let circ = load_tk1_json_reader(TK1.as_bytes())?;

    println!("{}", circ.dot_string());

    // This fails.
    let badger = DefaultBadgerOptimiser::default_with_rewriter_binary("test_files/nam_6_3.rwr")?;

    let mut new_circ = badger.optimise(&circ, Some(0), 1.try_into().unwrap(), false, 10);

    new_circ.update_validate(&REGISTRY)?;

    Ok(())
}
