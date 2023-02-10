use lazy_static::lazy_static;
use std::{
    any::Any,
    cmp::max,
    ops::{Add, Div, Mul, Neg, Sub},
};

use cgmath::num_traits::ToPrimitive;
use downcast_rs::{impl_downcast, Downcast};
use num_rational::Rational64;

#[cfg(feature = "pyo3")]
use pyo3::prelude::*;

pub(crate) type Param = f64;

#[cfg_attr(feature = "pyo3", pyclass)]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub enum WireType {
    Qubit,
    LinearBit,
    Bool,
    I32,
    F64,
    Quat64,
    Angle,
}

#[cfg_attr(feature = "pyo3", pyclass)]
#[derive(Clone)]
pub struct Signature {
    pub linear: Vec<WireType>,
    pub nonlinear: [Vec<WireType>; 2],
}

#[cfg_attr(feature = "pyo3", pymethods)]
impl Signature {
    pub fn len(&self) -> usize {
        self.linear.len() + max(self.nonlinear[0].len(), self.nonlinear[1].len())
    }

    pub fn is_empty(&self) -> bool {
        self.linear.is_empty() && self.nonlinear[0].is_empty() && self.nonlinear[1].is_empty()
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
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[cfg_attr(feature = "pyo3", pyclass(name = "Rational"))]
pub struct Rational(pub Rational64);

impl From<Rational64> for Rational {
    fn from(r: Rational64) -> Self {
        Self(r)
    }
}
// angle is contained value * pi in radians
#[derive(Clone, PartialEq, Debug, Copy)]
pub enum AngleValue {
    F64(f64),
    Rational(Rational),
}

impl AngleValue {
    fn binary_op<F: FnOnce(f64, f64) -> f64, G: FnOnce(Rational64, Rational64) -> Rational64>(
        self,
        rhs: Self,
        opf: F,
        opr: G,
    ) -> Self {
        match (self, rhs) {
            (AngleValue::F64(x), AngleValue::F64(y)) => AngleValue::F64(opf(x, y)),
            (AngleValue::F64(x), AngleValue::Rational(y))
            | (AngleValue::Rational(y), AngleValue::F64(x)) => {
                AngleValue::F64(opf(x, y.0.to_f64().unwrap()))
            }
            (AngleValue::Rational(x), AngleValue::Rational(y)) => {
                AngleValue::Rational(Rational(opr(x.0, y.0)))
            }
        }
    }

    fn unary_op<F: FnOnce(f64) -> f64, G: FnOnce(Rational64) -> Rational64>(
        self,
        opf: F,
        opr: G,
    ) -> Self {
        match self {
            AngleValue::F64(x) => AngleValue::F64(opf(x)),
            AngleValue::Rational(x) => AngleValue::Rational(Rational(opr(x.0))),
        }
    }

    pub fn to_f64(&self) -> f64 {
        match self {
            AngleValue::F64(x) => *x,
            AngleValue::Rational(x) => x.0.to_f64().expect("Floating point conversion error."),
        }
    }

    pub fn radians(&self) -> f64 {
        self.to_f64() * std::f64::consts::PI
    }
}

impl Add for AngleValue {
    type Output = AngleValue;

    fn add(self, rhs: Self) -> Self::Output {
        self.binary_op(rhs, |x, y| x + y, |x, y| x + y)
    }
}

impl Sub for AngleValue {
    type Output = AngleValue;

    fn sub(self, rhs: Self) -> Self::Output {
        self.binary_op(rhs, |x, y| x - y, |x, y| x - y)
    }
}

impl Mul for AngleValue {
    type Output = AngleValue;

    fn mul(self, rhs: Self) -> Self::Output {
        self.binary_op(rhs, |x, y| x * y, |x, y| x * y)
    }
}

impl Div for AngleValue {
    type Output = AngleValue;

    fn div(self, rhs: Self) -> Self::Output {
        self.binary_op(rhs, |x, y| x / y, |x, y| x / y)
    }
}

impl Neg for AngleValue {
    type Output = AngleValue;

    fn neg(self) -> Self::Output {
        self.unary_op(|x| -x, |x| -x)
    }
}

impl Add for &AngleValue {
    type Output = AngleValue;

    fn add(self, rhs: Self) -> Self::Output {
        self.binary_op(*rhs, |x, y| x + y, |x, y| x + y)
    }
}

impl Sub for &AngleValue {
    type Output = AngleValue;

    fn sub(self, rhs: Self) -> Self::Output {
        self.binary_op(*rhs, |x, y| x - y, |x, y| x - y)
    }
}

impl Mul for &AngleValue {
    type Output = AngleValue;

    fn mul(self, rhs: Self) -> Self::Output {
        self.binary_op(*rhs, |x, y| x * y, |x, y| x * y)
    }
}

impl Div for &AngleValue {
    type Output = AngleValue;

    fn div(self, rhs: Self) -> Self::Output {
        self.binary_op(*rhs, |x, y| x / y, |x, y| x / y)
    }
}

impl Neg for &AngleValue {
    type Output = AngleValue;

    fn neg(self) -> Self::Output {
        self.unary_op(|x| -x, |x| -x)
    }
}

#[derive(Clone, Copy, PartialEq, Debug)]
#[cfg_attr(feature = "pyo3", pyclass(name = "Quaternion"))]
pub struct Quat(pub cgmath::Quaternion<f64>);

impl From<cgmath::Quaternion<f64>> for Quat {
    fn from(q: cgmath::Quaternion<f64>) -> Self {
        Self(q)
    }
}

#[cfg_attr(feature = "pyo3", derive(FromPyObject))]
#[derive(Clone, PartialEq, Debug)]
pub enum ConstValue {
    Bool(bool),
    I32(i32),
    F64(f64),
    Angle(AngleValue),
    Quat64(Quat),
}

impl ConstValue {
    pub fn get_type(&self) -> WireType {
        match self {
            Self::Bool(_) => WireType::Bool,
            Self::I32(_) => WireType::I32,
            Self::F64(_) => WireType::F64,
            Self::Angle(_) => WireType::Angle,
            Self::Quat64(_) => WireType::Quat64,
        }
    }

    pub fn f64_angle(val: f64) -> Self {
        Self::Angle(AngleValue::F64(val))
    }
}

#[derive(Debug)]
pub struct ToCircuitFail;
pub trait CustomOp: Send + Sync + std::fmt::Debug + CustomBoxClone + Any + Downcast {
    fn signature(&self) -> Option<Signature>;

    fn to_circuit(&self) -> Result<super::circuit::Circuit, ToCircuitFail>;

    // fn as_any(&self) -> &dyn Any;
}

impl_downcast!(CustomOp);

pub trait CustomBoxClone {
    fn clone_box(&self) -> Box<dyn CustomOp>;

    // fn partial_cmp(&self, rhs)
}

impl<T> CustomBoxClone for T
where
    T: CustomOp + Clone,
{
    fn clone_box(&self) -> Box<dyn CustomOp> {
        Box::new(self.clone())
    }
}

impl Clone for Box<dyn CustomOp> {
    fn clone(&self) -> Box<dyn CustomOp> {
        self.clone_box()
    }
}

fn _deref_pair<'a, 'b, T: CustomOp>(
    a: &'a dyn CustomOp,
    b: &'b dyn CustomOp,
) -> (Option<&'a T>, Option<&'b T>) {
    (a.downcast_ref::<T>(), b.downcast_ref::<T>())
}

/// Attempt to downcast pairs to known custom types in turn and call their
/// equality methods (if implemented)
#[allow(unused_variables)]
fn custom_eq(x: &dyn CustomOp, y: &dyn CustomOp) -> bool {
    if let (Some(x), Some(y)) = _deref_pair::<tket_json_rs::circuit_json::Operation>(x, y) {
        return x == y;
    } else {
    }

    #[cfg(feature = "tkcxx")]
    if let (Some(x), Some(y)) = _deref_pair::<super::unitarybox::SU2>(x, y) {
        return x == y;
    } else {
    }

    #[cfg(feature = "pyo3")]
    if let (Some(x), Some(y)) = _deref_pair::<super::py_circuit::PyCustom>(x, y) {
        return x == y;
    } else {
    }
    false
}

#[derive(Clone, Debug)]
pub enum Op {
    H,
    CX,
    CZ,
    ZZMax,
    Reset,
    Input,
    Output,
    Noop(WireType),
    Measure,
    Barrier,
    AngleAdd,
    AngleMul,
    AngleNeg,
    QuatMul,
    Copy { n_copies: u32, typ: WireType },
    Const(ConstValue),
    RxF64,
    RzF64,
    TK1,
    Rotation,
    ToRotation,
    Swap,
    Custom(Box<dyn CustomOp>),
}

impl PartialEq for Op {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Noop(l0), Self::Noop(r0)) => l0 == r0,
            (
                Self::Copy {
                    n_copies: l_n_copies,
                    typ: l_typ,
                },
                Self::Copy {
                    n_copies: r_n_copies,
                    typ: r_typ,
                },
            ) => l_n_copies == r_n_copies && l_typ == r_typ,
            (Self::Const(l0), Self::Const(r0)) => l0 == r0,
            (Self::Custom(x), Self::Custom(y)) => custom_eq(&**x, &**y),
            _ => core::mem::discriminant(self) == core::mem::discriminant(other),
        }
    }
}

impl Default for Op {
    fn default() -> Self {
        Self::Noop(WireType::Qubit)
    }
}
lazy_static! {
    static ref ONEQBSIG: Signature = Signature::new_linear(vec![WireType::Qubit]);
}
lazy_static! {
    static ref TWOQBSIG: Signature = Signature::new_linear(vec![WireType::Qubit, WireType::Qubit]);
}

pub fn approx_eq(x: f64, y: f64, modulo: u32, tol: f64) -> bool {
    let modulo = f64::from(modulo);
    let x = (x - y) / modulo;

    let x = x - x.floor();

    let r = modulo * x;

    r < tol || r > modulo - tol
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
            Op::Noop(typ) => Signature::new_linear(vec![*typ]),
            Op::H | Op::Reset => ONEQBSIG.clone(),
            Op::CX | Op::ZZMax | Op::Swap | Op::CZ => TWOQBSIG.clone(),
            Op::Measure => Signature::new_linear(vec![WireType::Qubit, WireType::LinearBit]),
            Op::AngleAdd | Op::AngleMul => binary_op(WireType::Angle),
            Op::QuatMul => binary_op(WireType::Quat64),
            Op::AngleNeg => Signature::new_nonlinear(vec![WireType::Angle], vec![WireType::Angle]),
            Op::Copy { n_copies, typ } => {
                Signature::new_nonlinear(vec![*typ], vec![*typ; *n_copies as usize])
            }
            Op::Const(x) => Signature::new_nonlinear(vec![], vec![x.get_type()]),

            Op::RxF64 | Op::RzF64 => {
                Signature::new(vec![WireType::Qubit], [vec![WireType::Angle], vec![]])
            }
            Op::TK1 => Signature::new(vec![WireType::Qubit], [vec![WireType::Angle; 3], vec![]]),
            Op::Rotation => Signature::new(vec![WireType::Qubit], [vec![WireType::Quat64], vec![]]),
            Op::ToRotation => Signature::new_nonlinear(
                vec![WireType::Angle, WireType::F64, WireType::F64, WireType::F64],
                vec![WireType::Quat64],
            ),
            Op::Custom(x) => x.signature()?,
            _ => return None,
        })
    }

    pub fn get_params(&self) -> Vec<Param> {
        todo!()
    }
    // pub fn dagger(&self) -> Option<Self> {
    //     Some(match self {
    //         Op::Noop => Op::Noop,
    //         Op::H => Op::H,
    //         Op::CX => Op::CX,
    //         Op::ZZMax => Op::ZZPhase(Param::new("-0.5")),
    //         Op::TK1(a, b, c) => Op::TK1(c.neg(), b.neg(), a.neg()),
    //         Op::Rx(p) => Op::Rx(p.neg()),
    //         Op::Ry(p) => Op::Ry(p.neg()),
    //         Op::Rz(p) => Op::Rz(p.neg()),
    //         Op::ZZPhase(p) => Op::ZZPhase(p.neg()),
    //         Op::PhasedX(p1, p2) => Op::PhasedX(p1.neg(), p2.to_owned()),
    //         _ => return None,
    //     })
    // }

    // pub fn identity_up_to_phase(&self) -> Option<f64> {
    //     let two: Param = 2.0.into();
    //     match self {
    //         Op::Rx(p) | Op::Ry(p) | Op::Rz(p) | Op::ZZPhase(p) | Op::PhasedX(p, _) => {
    //             if equiv_0(p, 4) {
    //                 Some(0.0)
    //             } else if equiv_0(&(p + two), 2) {
    //                 Some(1.0)
    //             } else {
    //                 None
    //             }
    //         }
    //         Op::TK1(a, b, c) => {
    //             let s = a + c;
    //             if equiv_0(&s, 2) && equiv_0(b, 2) {
    //                 Some(if equiv_0(&s, 4) ^ equiv_0(b, 4) {
    //                     1.0
    //                 } else {
    //                     0.0
    //                 })
    //             } else {
    //                 None
    //             }
    //         }
    //         Op::Noop => Some(0.0),
    //         _ => None,
    //     }
    // }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "pyo3")]
    fn py_int(i: i32) -> Op {
        use crate::circuit::py_circuit::PyCustom;
        Op::Custom(Box::new(PyCustom(Python::with_gil(|py| i.into_py(py)))))
    }

    #[cfg(feature = "tkcxx")]
    fn unitary_x() -> Op {
        use num_complex::Complex;

        use crate::circuit::unitarybox::SU2;

        Op::Custom(Box::new(SU2([
            [Complex::new(0.0, 0.0), Complex::new(1.0, 0.0)],
            [Complex::new(1.0, 0.0), Complex::new(0.0, 0.0)],
        ])))
    }

    #[cfg(feature = "tkcxx")]
    fn unitary_z() -> Op {
        use num_complex::Complex;

        use crate::circuit::unitarybox::SU2;

        Op::Custom(Box::new(SU2([
            [Complex::new(1.0, 0.0), Complex::new(0.0, 0.0)],
            [Complex::new(0.0, 0.0), Complex::new(-1.0, 0.0)],
        ])))
    }
    #[test]
    fn equality() {
        #[cfg(feature = "pyo3")]
        pyo3::prepare_freethreaded_python();
        let ops = [
            Op::Input,
            Op::Output,
            #[cfg(feature = "pyo3")]
            py_int(123),
            #[cfg(feature = "pyo3")]
            py_int(321),
            #[cfg(feature = "tkcxx")]
            unitary_x(),
            #[cfg(feature = "tkcxx")]
            unitary_z(),
            Op::Copy {
                n_copies: 3,
                typ: WireType::Qubit,
            },
        ];

        for o in &ops {
            assert_eq!(o, &o.clone());
        }

        for window in ops.windows(2) {
            assert!(window[0] != window[1]);
        }
    }
}
