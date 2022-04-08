pub mod circuit;
mod graph;

pub mod json;
pub mod passes;

#[cfg(test)]
mod tests {
    use crate::{
        circuit::{
            circuit::{Circuit, UnitID},
            operation::{equiv_0, ConstValue, Op},
            operation::{Param, WireType},
        },
        graph::dot::dot_string,
        graph::graph::{NodePort, PortIndex},
        json::circuit_json::{self, SerialCircuit},
        passes::classical::find_const_ops,
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
    fn test_const_fold() {
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

        let rewrite = find_const_ops(&circ).next().unwrap();

        circ.dag.apply_rewrite(rewrite).unwrap();

        // println!("{}", dot_string(&circ.dag));
        assert_eq!(circ.dag.node_count(), 3);
        assert_eq!(circ.dag.edge_count(), 1);
        let mut nodeit = circ.dag.nodes();
        // skip input and output
        nodeit.next();
        nodeit.next();

        assert_eq!(
            &circ.dag.node_weight(nodeit.next().unwrap()).unwrap().op,
            &Op::Const(ConstValue::F64(2.0))
        );

        // TODO test repeated application, negation, !copy!
    }
}
