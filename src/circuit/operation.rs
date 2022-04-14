#![allow(dead_code)]

use lazy_static::lazy_static;
use std::cmp::max;

use cgmath::Quaternion;
use symengine::Expression;
pub(crate) type Param = Expression;

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub enum WireType {
    Qubit,
    LinearBit,
    Bool,
    I32,
    F64,
    Quat64,
}
#[derive(Clone)]
pub struct Signature {
    pub linear: Vec<WireType>,
    pub nonlinear: [Vec<WireType>; 2],
}

impl Signature {
    pub fn new(linear: Vec<WireType>, nonlinear: [Vec<WireType>; 2]) -> Self {
        Self { linear, nonlinear }
    }
    pub fn new_linear(linear: Vec<WireType>) -> Self {
        Self {
            linear,
            nonlinear: [vec![], vec![]],
        }
    }

    pub fn new_nonlinear(inputs: Vec<WireType>, outputs: Vec<WireType>) -> Self {
        Self {
            linear: vec![],
            nonlinear: [inputs, outputs],
        }
    }

    pub fn len(&self) -> usize {
        self.linear.len() + max(self.nonlinear[0].len(), self.nonlinear[1].len())
    }

    pub fn purely_linear(&self) -> bool {
        self.nonlinear[0].is_empty() && self.nonlinear[1].is_empty()
    }

    pub fn purely_classical(&self) -> bool {
        !self
            .linear
            .iter()
            .chain(self.nonlinear[0].iter())
            .chain(self.nonlinear[1].iter())
            .any(|typ| matches!(typ, WireType::Qubit))
    }
}

#[derive(Clone, PartialEq, Debug)]
pub enum ConstValue {
    Bool(bool),
    I32(i32),
    F64(f64),
    Quat64(Quaternion<f64>),
}

impl ConstValue {
    pub fn get_type(&self) -> WireType {
        match self {
            Self::Bool(_) => WireType::Bool,
            Self::I32(_) => WireType::I32,
            Self::F64(_) => WireType::F64,
            Self::Quat64(_) => WireType::Quat64,
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
    FAdd,
    FMul,
    FNeg,
    QuatMul,
    // Sin,
    // Cos,
    Copy { n_copies: u32, typ: WireType },
    Const(ConstValue),
    RxF64,
    RzF64,
    Rotation,
    ToRotation,
}

impl Default for Op {
    fn default() -> Self {
        Self::Noop
    }
}
lazy_static! {
    static ref ONEQBSIG: Signature = Signature::new_linear(vec![WireType::Qubit]);
}
lazy_static! {
    static ref TWOQBSIG: Signature = Signature::new_linear(vec![WireType::Qubit, WireType::Qubit]);
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

fn binary_op(typ: WireType) -> Signature {
    Signature::new_nonlinear(vec![typ, typ], vec![typ])
}

impl Op {
    pub fn is_one_qb_gate(&self) -> bool {
        self.signature()
            .map_or(false, |sig| matches!(&sig.linear[..], &[WireType::Qubit]))
    }

    pub fn is_two_qb_gate(&self) -> bool {
        self.signature().map_or(false, |sig| {
            matches!(&sig.linear[..], &[WireType::Qubit, WireType::Qubit])
        })
    }

    pub fn is_pure_classical(&self) -> bool {
        self.signature().map_or(false, |x| x.purely_classical())
    }

    pub fn signature(&self) -> Option<Signature> {
        Some(match self {
            Op::H
            | Op::Reset
            | Op::Rx(_)
            | Op::Ry(_)
            | Op::Rz(_)
            | Op::TK1(..)
            | Op::PhasedX(..) => ONEQBSIG.clone(),
            Op::CX | Op::ZZMax | Op::ZZPhase(..) => TWOQBSIG.clone(),
            Op::Measure => Signature::new_linear(vec![WireType::Qubit, WireType::LinearBit]),
            Op::FAdd | Op::FMul => binary_op(WireType::F64),
            Op::QuatMul => binary_op(WireType::Quat64),
            Op::FNeg => Signature::new_nonlinear(vec![WireType::F64], vec![WireType::F64]),
            Op::Copy { n_copies, typ } => {
                let typ = typ.clone();
                Signature::new_nonlinear(vec![typ.clone()], vec![typ; *n_copies as usize])
            }
            Op::Const(x) => Signature::new_nonlinear(vec![], vec![x.get_type()]),

            Op::RxF64 | Op::RzF64 => {
                Signature::new(vec![WireType::Qubit], [vec![WireType::F64], vec![]])
            }
            Op::Rotation => Signature::new(vec![WireType::Qubit], [vec![WireType::Quat64], vec![]]),
            Op::ToRotation => Signature::new_nonlinear(
                vec![WireType::F64, WireType::F64, WireType::F64, WireType::F64],
                vec![WireType::Quat64],
            ),
            _ => return None,
        })
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
            Op::Noop => Some(0.0),
            _ => None,
        }
    }
}
