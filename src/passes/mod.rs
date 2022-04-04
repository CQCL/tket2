pub mod redundancy;

#[cfg(test)]
mod tests {
    use symengine::Expression;

    use crate::{
        circuit::circuit::Circuit,
        circuit_json::{self, SerialCircuit},
    };

    use super::redundancy::remove_redundancies;

    #[test]
    fn test_remove_redundancies() {
        // circuit with only redundant gates; identity unitary
        //[Rz(a) q[0];, Rz(-a) q[0];, CX q[0], q[1];, CX q[0], q[1];, Rx(2) q[1];]
        let circ_s = r#"{"bits": [], "commands": [{"args": [["q", [0]]], "op": {"params": ["a"], "type": "Rz"}}, {"args": [["q", [0]]], "op": {"params": ["-a"], "type": "Rz"}}, {"args": [["q", [0]], ["q", [1]]], "op": {"type": "CX"}}, {"args": [["q", [0]], ["q", [1]]], "op": {"type": "CX"}}, {"args": [["q", [1]]], "op": {"params": ["2.0"], "type": "Rx"}}], "implicit_permutation": [[["q", [0]], ["q", [0]]], [["q", [1]], ["q", [1]]]], "phase": "0.0", "qubits": [["q", [0]], ["q", [1]]]}"#;
        let ser: circuit_json::SerialCircuit = serde_json::from_str(circ_s).unwrap();

        let circ: Circuit = ser.clone().into();

        let circ2 = remove_redundancies(circ);
        let _reser: SerialCircuit = circ2.into();

        assert_eq!(_reser.commands.len(), 0);
        // Rx(2pi) introduces a phase
        assert_eq!(_reser.phase, Expression::new("1.0"));
    }
}
