pub mod circuit;
mod graph;

pub mod json;
pub mod passes;

#[cfg(test)]
mod tests {
    use crate::{
        circuit::{
            circuit::{Circuit, UnitID},
            dag::Vertex,
            operation::{equiv_0, ConstValue, Op},
            operation::{Param, WireType},
        },
        graph::dot::dot_string,
        graph::graph::{Direction, NodePort, PortIndex},
        json::circuit_json::{self, SerialCircuit},
    };
    fn eval_const(circ: &Circuit, n: Vertex) -> Vec<Option<Op>> {
        let op = &circ.dag.node_weight(n).unwrap().op;
        let mut inputs = circ
            .dag
            .neighbours(n, Direction::Incoming)
            .map(|np| eval_const(circ, np.node).remove(np.port.index()));

        let failed_return = vec![None; circ.dag.node_edges(n, Direction::Outgoing).count()];

        match op {
            Op::Const(ConstValue::F64(_)) => vec![Some(op.clone())],
            Op::Copy {
                n_copies,
                typ: WireType::F64,
            } => vec![inputs.next().expect("Input missing"); *n_copies as usize],
            Op::FAdd => match &inputs.take(2).collect::<Vec<_>>()[..] {
                [Some(Op::Const(ConstValue::F64(x))), Some(Op::Const(ConstValue::F64(y)))] => {
                    vec![Some(Op::Const(ConstValue::F64(x + y)))]
                }
                _ => failed_return,
            },
            _ => failed_return,
        }
    }

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
        assert_eq!(Param::new("0.1").neg(), Param::new("-0.1"));

        assert!(Param::new("x").eval().is_none());
        assert_eq!(Param::new("2.0 + 2.0/4").eval(), Some(2.5));
        assert!(equiv_0(&Param::new("0"), 4));
        assert!(equiv_0(&Param::new("4.0"), 4));
        assert!(equiv_0(&Param::new("8.0"), 4));
        assert!(!equiv_0(&Param::new("2.0"), 4));
        assert!(equiv_0(&Param::new("2.0"), 2));
        assert!(!equiv_0(&Param::new("0.5"), 2));
    }

    #[test]
    fn test_dagger() {
        assert_eq!(Op::H.dagger().unwrap(), Op::H);
        assert_eq!(Op::CX.dagger().unwrap(), Op::CX);
        assert_eq!(Op::Rx(0.1.into()).dagger().unwrap(), Op::Rx((-0.1).into()));
        assert_eq!(
            Op::Rz(Param::new("x")).dagger().unwrap(),
            Op::Rz(Param::new("-x"))
        );
    }

    #[test]
    fn test_fadd() {
        let mut circ = Circuit::new();

        circ.add_unitid(UnitID::F64("a".into()));
        circ.add_unitid(UnitID::F64("b".into()));
        let [input, output] = circ.boundary();

        let fadd = circ.add_vertex(Op::FAdd);
        circ.add_edge(
            NodePort::new(input, PortIndex::new(0)),
            NodePort::new(fadd, PortIndex::new(0)),
            WireType::F64,
        );

        circ.add_edge(
            NodePort::new(input, PortIndex::new(1)),
            NodePort::new(fadd, PortIndex::new(1)),
            WireType::F64,
        );

        circ.add_edge(
            NodePort::new(fadd, PortIndex::new(0)),
            NodePort::new(output, PortIndex::new(0)),
            WireType::F64,
        );
    }

    #[test]
    fn test_copy() {
        let mut circ = Circuit::new();

        circ.add_unitid(UnitID::F64("a".into()));
        let [input, output] = circ.boundary();

        let fadd = circ.add_vertex(Op::FAdd);
        let e = circ.add_edge(
            NodePort::new(input, PortIndex::new(0)),
            NodePort::new(fadd, PortIndex::new(0)),
            WireType::F64,
        );

        let copy = circ.copy_edge(e, 2).unwrap();

        circ.add_edge(
            NodePort::new(copy, PortIndex::new(1)),
            NodePort::new(fadd, PortIndex::new(1)),
            WireType::F64,
        );

        circ.add_edge(
            NodePort::new(fadd, PortIndex::new(0)),
            NodePort::new(output, PortIndex::new(0)),
            WireType::F64,
        );
        println!("{}", dot_string(&circ.dag));
    }

    #[test]
    fn test_eval() {
        let mut circ = Circuit::new();

        let [_, output] = circ.boundary();

        let fadd = circ.add_vertex(Op::FAdd);
        let one = circ.add_vertex(Op::Const(ConstValue::F64(0.5)));
        let two = circ.add_vertex(Op::Const(ConstValue::F64(1.5)));
        let _e1 = circ.add_edge(
            NodePort::new(one, PortIndex::new(0)),
            NodePort::new(fadd, PortIndex::new(0)),
            WireType::F64,
        );
        let _e2 = circ.add_edge(
            NodePort::new(two, PortIndex::new(0)),
            NodePort::new(fadd, PortIndex::new(1)),
            WireType::F64,
        );

        let _out = circ.add_edge(
            NodePort::new(fadd, PortIndex::new(0)),
            NodePort::new(output, PortIndex::new(0)),
            WireType::F64,
        );
        assert!(match &eval_const(&circ, fadd)[..] {
            [Some(Op::Const(ConstValue::F64(x)))] => *x == 2.0,
            _ => false,
        })
    }
}
