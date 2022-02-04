mod circuit_json;
mod optype;

#[cfg(test)]
mod tests {
    use crate::circuit_json;
    #[test]
    fn it_works() {
        let circ_s = r#"{"bits": [["c", [0]], ["c", [1]]], "commands": [{"args": [["q", [0]]], "op": {"type": "H"}}, {"args": [["q", [0]], ["q", [1]]], "op": {"type": "CX"}}, {"args": [["q", [0]], ["c", [0]]], "op": {"type": "Measure"}}, {"args": [["q", [1]], ["c", [1]]], "op": {"type": "Measure"}}], "implicit_permutation": [[["q", [0]], ["q", [0]]], [["q", [1]], ["q", [1]]]], "phase": "0.0", "qubits": [["q", [0]], ["q", [1]]]}"#;
        let p: circuit_json::SerialCircuit = serde_json::from_str(circ_s).unwrap();
        assert_eq!(p.commands.len(), 4);
    }
}
