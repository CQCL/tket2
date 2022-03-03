#![allow(dead_code)]

use std::cmp::max;

use lazy_static::lazy_static;

use crate::{circuit_json::Operation, optype::OpType};
// use symengine::Expression;
pub(crate) type Param = String;

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum WireType {
    Quantum,
    Classical,
    Bool,
}
#[derive(Clone)]
pub enum Signature {
    Linear(Vec<WireType>),
    NonLinear(Vec<WireType>, Vec<WireType>),
}

impl Signature {
    pub fn len(&self) -> usize {
        match self {
            Signature::Linear(s) => s.len(),
            Signature::NonLinear(s1, s2) => max(s1.len(), s2.len()),
        }
    }
}

#[derive(Clone)]
pub enum Op {
    H,
    CX,
    ZZMax,
    Reset,
    Input,
    Output,
    Rx(Param),
    Ry(Param),
    Rz(Param),
    ZZPhase(Param, Param),
    Measure,
    Barrier,
}

lazy_static! {
    static ref ONEQBSIG: Signature = Signature::Linear(vec![WireType::Quantum]);
}
lazy_static! {
    static ref TWOQBSIG: Signature = Signature::Linear(vec![WireType::Quantum, WireType::Quantum]);
}

impl Op {
    fn is_one_qb_gate(&self) -> bool {
        match self.signature() {
            Signature::Linear(v) => matches!(&v[..], &[WireType::Quantum]),
            _ => false,
        }
    }

    fn is_two_qb_gate(&self) -> bool {
        match self.signature() {
            Signature::Linear(v) => matches!(&v[..], &[WireType::Quantum, WireType::Quantum]),
            _ => false,
        }
    }

    pub fn signature(&self) -> Signature {
        match self {
            Op::H | Op::Reset | Op::Rx(_) | Op::Ry(_) | Op::Rz(_) => ONEQBSIG.clone(),
            Op::CX | Op::ZZMax | Op::ZZPhase(..) => TWOQBSIG.clone(),
            Op::Measure => Signature::Linear(vec![WireType::Quantum, WireType::Classical]),
            _ => panic!("Gate signature unknwon."),
        }
    }

    pub fn get_params(&self) -> Vec<Param> {
        todo!()
    }

    pub fn to_serialized(&self) -> Operation {
        let (op_type, params) = match self {
            Op::H => (OpType::H, vec![]),
            Op::CX => (OpType::CX, vec![]),
            Op::ZZMax => (OpType::ZZMax, vec![]),
            Op::Reset => (OpType::Reset, vec![]),
            Op::Input => (OpType::Input, vec![]),
            Op::Output => (OpType::Output, vec![]),
            Op::Rx(p) => (OpType::Rx, vec![p]),
            Op::Ry(p) => (OpType::Ry, vec![p]),
            Op::Rz(p) => (OpType::Rz, vec![p]),
            Op::ZZPhase(p1, p2) => (OpType::ZZPhase, vec![p1, p2]),
            Op::Measure => (OpType::Measure, vec![]),
            Op::Barrier => (OpType::Barrier, vec![]),
        };
        // let signature = match self.signature() {
        //     Signature::Linear(sig) => sig.iter().map(|wt| match wt {
        //         WireType::Quantum => todo!(),
        //         WireType::Classical => todo!(),
        //         WireType::Bool => todo!(),
        //     }),
        //     Signature::NonLinear(_, _) => panic!(),
        // }
        let params = if params.is_empty() {
            None
        } else {
            Some(params.iter().map(|&s| s.clone()).collect())
        };
        Operation {
            op_type,
            params,
            signature: None,
            op_box: None,
            n_qb: None,
            conditional: None,
        }
    }
}
