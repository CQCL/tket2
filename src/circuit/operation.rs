#![allow(dead_code)]

use std::{cmp::max, rc::Rc};

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
pub trait Op {
    // pub trait Op: OpClone {
    fn signature(&self) -> Signature;

    fn get_params(&self) -> Vec<Param>;

    fn to_serialized(&self) -> Operation;
}

pub(crate) type OpPtr = Rc<dyn Op>;

#[derive(Clone)]
pub enum GateOp {
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
}

impl GateOp {
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
}

lazy_static! {
    static ref ONEQBSIG: Signature = Signature::Linear(vec![WireType::Quantum]);
}
lazy_static! {
    static ref TWOQBSIG: Signature = Signature::Linear(vec![WireType::Quantum, WireType::Quantum]);
}

impl Op for GateOp {
    fn signature(&self) -> Signature {
        match self {
            GateOp::H | GateOp::Reset | GateOp::Rx(_) | GateOp::Ry(_) | GateOp::Rz(_) => {
                ONEQBSIG.clone()
            }
            GateOp::CX | GateOp::ZZMax | GateOp::ZZPhase(..) => TWOQBSIG.clone(),
            GateOp::Measure => Signature::Linear(vec![WireType::Quantum, WireType::Classical]),
            _ => panic!("Gate signature unknwon."),
        }
    }

    fn get_params(&self) -> Vec<Param> {
        todo!()
    }

    fn to_serialized(&self) -> Operation {
        let (op_type, params) = match self {
            GateOp::H => (OpType::H, vec![]),
            GateOp::CX => (OpType::CX, vec![]),
            GateOp::ZZMax => (OpType::ZZMax, vec![]),
            GateOp::Reset => (OpType::Reset, vec![]),
            GateOp::Input => (OpType::Input, vec![]),
            GateOp::Output => (OpType::Output, vec![]),
            GateOp::Rx(p) => (OpType::Rx, vec![p]),
            GateOp::Ry(p) => (OpType::Ry, vec![p]),
            GateOp::Rz(p) => (OpType::Rz, vec![p]),
            GateOp::ZZPhase(p1, p2) => (OpType::ZZPhase, vec![p1, p2]),
            GateOp::Measure => (OpType::Measure, vec![]),
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

pub enum MetaOp {
    Barrier,
}

pub enum ClassicalOp {
    And,
    Xor,
    Or,
}
