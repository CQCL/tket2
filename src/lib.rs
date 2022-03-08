mod circuit;
mod circuit_json;
mod graph;
mod json_convert;
mod optype;

#[cfg(test)]
mod tests {
    use crate::{
        circuit::{circuit::Circuit, operation::Param},
        circuit_json::{self, SerialCircuit},
    };
    #[test]
    fn read_json() {
        // let expr = symengine::Expression::new("a + b + 3");
        let circ_s = r#"{"bits": [["c", [0]], ["c", [1]]], "commands": [{"args": [["q", [0]]], "op": {"type": "H"}}, {"args": [["q", [0]], ["q", [1]]], "op": {"type": "CX"}}, {"args": [["q", [0]], ["c", [0]]], "op": {"type": "Measure"}}, {"args": [["q", [1]], ["c", [1]]], "op": {"type": "Measure"}}], "implicit_permutation": [[["q", [0]], ["q", [0]]], [["q", [1]], ["q", [1]]]], "phase": "0.0", "qubits": [["q", [0]], ["q", [1]]]}"#;
        let ser: circuit_json::SerialCircuit = serde_json::from_str(circ_s).unwrap();
        assert_eq!(ser.commands.len(), 4);

        let circ: Circuit = ser.clone().into();

        let _reser: SerialCircuit = circ.into();

        assert_eq!(&ser, &_reser);
        // ser and reser cannot be compared because they will be different up to
        // topsort ordering of parallel commands
    }

    #[test]
    fn test_param() {
        assert_eq!(Param::new("3") + Param::new("x"), Param::new("3 + x"));
        assert_eq!(Param::new("0") - Param::new("0.1"), Param::new("-0.1"));
    }
}
