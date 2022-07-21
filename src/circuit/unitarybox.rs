use super::operation::CustomOp;
use super::operation::Signature;
use super::operation::WireType;
use num_complex::Complex;
use tket_json_rs::circuit_json;
use tket_rs::make_box;

#[derive(Debug, Clone, PartialEq)]
pub struct SU2(pub [[Complex<f64>; 2]; 2]);

impl CustomOp for SU2 {
    fn signature(&self) -> Option<Signature> {
        Some(Signature::new_linear(vec![WireType::Qubit]))
    }

    fn to_circuit(&self) -> Result<super::circuit::Circuit, super::operation::ToCircuitFail> {
        let mut arr: [[[f64; 2]; 2]; 2] = Default::default();
        for (i, row) in self.0.into_iter().enumerate() {
            for (j, c) in row.into_iter().enumerate() {
                arr[i][j] = [c.re, c.im];
            }
        }

        let box2 = make_box(arr);
        let cj = box2.circ_json();

        let ser: circuit_json::SerialCircuit = serde_json::from_str(&cj).unwrap();

        Ok(ser.into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        circuit::{
            circuit::{Circuit, UnitID},
            operation::{AngleValue, ConstValue, Op},
        },
        graph::graph::PortIndex,
        passes::decompose_custom_pass,
        validate::check_soundness,
    };
    #[test]
    fn test_decompose_unitary() {
        let qubits = vec![UnitID::Qubit {
            reg_name: "q".into(),
            index: vec![0],
        }];
        let mut circ = Circuit::with_uids(qubits);
        let x_su2 = SU2([
            [Complex::new(0.0, 0.0), Complex::new(1.0, 0.0)],
            [Complex::new(1.0, 0.0), Complex::new(0.0, 0.0)],
        ]);
        circ.append_op(Op::Custom(Box::new(x_su2)), &vec![PortIndex::new(0)])
            .unwrap();
        check_soundness(&circ).unwrap();

        let (circ, success) = decompose_custom_pass(circ);
        println!("{}", circ.dot_string());
        assert!(success);
        check_soundness(&circ).unwrap();

        assert_eq!(circ.dag.node_count(), 6);
        for op in circ.dag.node_weights() {
            assert!(matches!(
                op.op,
                Op::Input | Op::Output | Op::Const(ConstValue::Angle(AngleValue::F64(_))) | Op::TK1
            ))
        }
    }
}
