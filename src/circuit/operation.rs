#![allow(dead_code)]

use lazy_static::lazy_static;
use std::cmp::max;
use symengine::Expression;

// use symengine::Expression;
pub(crate) type Param = Expression;

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

#[derive(Clone, PartialEq, Debug)]
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
    ZZPhase(Param),
    PhasedX(Param, Param),
    Measure,
    Barrier,
}

lazy_static! {
    static ref ONEQBSIG: Signature = Signature::Linear(vec![WireType::Quantum]);
}
lazy_static! {
    static ref TWOQBSIG: Signature = Signature::Linear(vec![WireType::Quantum, WireType::Quantum]);
}

fn neg_param(p: Param) -> Param {
    Param::new("0") - p
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
    pub fn dagger(&self) -> Option<Self> {
        Some(match self {
            Op::H => Op::H,
            Op::CX => Op::CX,
            Op::ZZMax => Op::ZZPhase(Param::new("-0.5")),

            Op::Rx(p) => Op::Rx(neg_param(p.to_owned())),
            Op::Ry(p) => Op::Ry(neg_param(p.to_owned())),
            Op::Rz(p) => Op::Rz(neg_param(p.to_owned())),
            Op::ZZPhase(p) => Op::ZZPhase(neg_param(p.to_owned())),
            Op::PhasedX(p1, p2) => Op::PhasedX(neg_param(p1.to_owned()), p2.to_owned()),
            _ => return None,
        })
    }
}
