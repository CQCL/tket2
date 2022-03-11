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
    Noop,
    Rx(Param),
    Ry(Param),
    Rz(Param),
    TK1(Param, Param, Param),
    ZZPhase(Param),
    PhasedX(Param, Param),
    Measure,
    Barrier,
}

impl Default for Op {
    fn default() -> Self {
        Self::Noop
    }
}
lazy_static! {
    static ref ONEQBSIG: Signature = Signature::Linear(vec![WireType::Quantum]);
}
lazy_static! {
    static ref TWOQBSIG: Signature = Signature::Linear(vec![WireType::Quantum, WireType::Quantum]);
}

pub fn approx_eq(x: f64, y: f64, modulo: u32, tol: f64) -> bool {
    let modulo = modulo as f64;
    let x = (x - y) / modulo;

    let x = x - x.floor();

    let r = modulo * x;

    r < tol || r > modulo - tol
}

pub fn equiv_0(p: &Param, modulo: u32) -> bool {
    if let Some(x) = p.eval() {
        approx_eq(x, 0.0, modulo, 1e-11)
    } else {
        false
    }
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
            Op::H
            | Op::Reset
            | Op::Rx(_)
            | Op::Ry(_)
            | Op::Rz(_)
            | Op::TK1(..)
            | Op::PhasedX(..) => ONEQBSIG.clone(),
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
            Op::TK1(a, b, c) => Op::TK1(c.neg(), b.neg(), a.neg()),
            Op::Rx(p) => Op::Rx(p.neg()),
            Op::Ry(p) => Op::Ry(p.neg()),
            Op::Rz(p) => Op::Rz(p.neg()),
            Op::ZZPhase(p) => Op::ZZPhase(p.neg()),
            Op::PhasedX(p1, p2) => Op::PhasedX(p1.neg(), p2.to_owned()),
            _ => return None,
        })
    }

    pub fn identity_up_to_phase(&self) -> Option<f64> {
        let two: Param = 2.0.into();
        match self {
            Op::Rx(p) | Op::Ry(p) | Op::Rz(p) | Op::ZZPhase(p) | Op::PhasedX(p, _) => {
                if equiv_0(p, 4) {
                    Some(0.0)
                } else if equiv_0(&(p + two), 2) {
                    Some(1.0)
                } else {
                    None
                }
            }
            Op::TK1(a, b, c) => {
                let s = a + c;
                if equiv_0(&s, 2) && equiv_0(b, 2) {
                    Some(if equiv_0(&s, 4) ^ equiv_0(b, 4) {
                        1.0
                    } else {
                        0.0
                    })
                } else {
                    None
                }
            }
            _ => None,
        }
    }
}
