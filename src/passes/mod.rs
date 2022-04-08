pub mod redundancy;

#[cfg(test)]
mod tests {
    use symengine::Expression;

    use crate::{
        circuit::{
            circuit::{Circuit, UnitID},
            operation::{Op, Param},
        },
        graph::{dot::dot_string, graph::PortIndex},
        json::circuit_json::{self, SerialCircuit},
    };

    use super::redundancy::remove_redundancies;

    #[test]
    fn test_remove_redundancies() {
        // circuit with only redundant gates; identity unitary
        //[Rz(a) q[0];, Rz(-a) q[0];, CX q[0], q[1];, CX q[0], q[1];, Rx(2) q[1];]
        let qubits = vec![
            UnitID::Qubit {
                name: "q".into(),
                index: vec![0],
            },
            UnitID::Qubit {
                name: "q".into(),
                index: vec![0],
            },
        ];
        let mut circ = Circuit::with_uids(qubits);

        circ.append_op(Op::Rz(Param::new("a")), &vec![PortIndex::new(0)])
            .unwrap();
        circ.append_op(Op::Rz(Param::new("-a")), &vec![PortIndex::new(0)])
            .unwrap();
        circ.append_op(Op::CX, &vec![PortIndex::new(0), PortIndex::new(1)])
            .unwrap();
        circ.append_op(Op::CX, &vec![PortIndex::new(0), PortIndex::new(1)])
            .unwrap();
        circ.append_op(Op::Rx(Param::new("2.0")), &vec![PortIndex::new(1)])
            .unwrap();

        println!("{}", dot_string(&circ.dag));

        let circ2 = remove_redundancies(circ);
        println!("{}", dot_string(&circ2.dag));

        let _reser: SerialCircuit = circ2.into();

        assert_eq!(_reser.commands.len(), 0);
        // Rx(2pi) introduces a phase
        assert_eq!(_reser.phase, Expression::new("1.0"));
    }
}
