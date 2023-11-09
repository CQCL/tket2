use std::collections::HashMap;

use crate::extension::{
    SYM_EXPR_T, SYM_OP_ID, TKET2_EXTENSION as EXTENSION, TKET2_EXTENSION_ID as EXTENSION_ID,
};
use hugr::{
    extension::{
        prelude::{BOOL_T, QB_T},
        ExtensionBuildError, ExtensionId, OpDef,
    },
    ops::{custom::ExternalOp, LeafOp, OpType},
    std_extensions::arithmetic::float_types::FLOAT64_TYPE,
    type_row,
    types::{
        type_param::{CustomTypeArg, TypeArg},
        FunctionType,
    },
    Extension,
};

use serde::{Deserialize, Serialize};

use std::str::FromStr;
use strum::IntoEnumIterator;
use strum_macros::{Display, EnumIter, EnumString, IntoStaticStr};
use thiserror::Error;

#[cfg(feature = "pyo3")]
use pyo3::pyclass;

use crate::extension::REGISTRY;

#[derive(
    Clone,
    Copy,
    Debug,
    Serialize,
    Deserialize,
    Hash,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    EnumIter,
    IntoStaticStr,
    EnumString,
)]
#[cfg_attr(feature = "pyo3", pyclass)]
#[allow(missing_docs)]
#[non_exhaustive]
/// Simple enum of tket 2 quantum operations.
pub enum T2Op {
    H,
    CX,
    T,
    S,
    X,
    Y,
    Z,
    Tdg,
    Sdg,
    ZZMax,
    Measure,
    RzF64,
    RxF64,
    PhasedX,
    ZZPhase,
    AngleAdd,
    CZ,
    TK1,
}

/// Whether an op is a given T2Op.
pub fn op_matches(op: &OpType, t2op: T2Op) -> bool {
    let Ok(op) = T2Op::try_from(op) else {
        return false;
    };
    op == t2op
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, EnumIter, Display, PartialEq, PartialOrd)]
#[cfg_attr(feature = "pyo3", pyclass)]
#[allow(missing_docs)]
/// Simple enum representation of Pauli matrices.
pub enum Pauli {
    I,
    X,
    Y,
    Z,
}

#[derive(Debug, Error, PartialEq, Clone, Copy)]
#[error("Not a T2Op.")]
pub struct NotT2Op;

// this trait could be implemented in Hugr
pub(crate) trait SimpleOpEnum:
    Into<&'static str> + FromStr + Copy + IntoEnumIterator
{
    type LoadError: std::error::Error;

    fn signature(&self) -> FunctionType;
    fn name(&self) -> &str {
        (*self).into()
    }
    fn from_extension_name(extension: &ExtensionId, op_name: &str)
        -> Result<Self, Self::LoadError>;
    fn try_from_op_def(op_def: &OpDef) -> Result<Self, Self::LoadError> {
        Self::from_extension_name(op_def.extension(), op_def.name())
    }
    fn add_to_extension<'e>(
        &self,
        ext: &'e mut Extension,
    ) -> Result<&'e OpDef, ExtensionBuildError>;

    fn all_variants() -> <Self as IntoEnumIterator>::Iterator {
        <Self as IntoEnumIterator>::iter()
    }
}

fn from_extension_name<T: SimpleOpEnum>(
    extension: &ExtensionId,
    op_name: &str,
) -> Result<T, NotT2Op> {
    if extension != &EXTENSION_ID {
        return Err(NotT2Op);
    }
    T::from_str(op_name).map_err(|_| NotT2Op)
}

impl Pauli {
    /// Check if this pauli commutes with another.
    pub fn commutes_with(&self, other: Self) -> bool {
        *self == Pauli::I || other == Pauli::I || *self == other
    }
}
impl SimpleOpEnum for T2Op {
    type LoadError = NotT2Op;
    fn signature(&self) -> FunctionType {
        use T2Op::*;
        let one_qb_row = type_row![QB_T];
        let two_qb_row = type_row![QB_T, QB_T];
        match self {
            H | T | S | X | Y | Z | Tdg | Sdg => FunctionType::new(one_qb_row.clone(), one_qb_row),
            CX | ZZMax | CZ => FunctionType::new(two_qb_row.clone(), two_qb_row),
            ZZPhase => FunctionType::new(type_row![QB_T, QB_T, FLOAT64_TYPE], two_qb_row),
            Measure => FunctionType::new(one_qb_row, type_row![QB_T, BOOL_T]),
            RzF64 | RxF64 => FunctionType::new(type_row![QB_T, FLOAT64_TYPE], one_qb_row),
            PhasedX => FunctionType::new(type_row![QB_T, FLOAT64_TYPE, FLOAT64_TYPE], one_qb_row),
            AngleAdd => FunctionType::new(
                type_row![FLOAT64_TYPE, FLOAT64_TYPE],
                type_row![FLOAT64_TYPE],
            ),
            TK1 => FunctionType::new(
                type_row![QB_T, FLOAT64_TYPE, FLOAT64_TYPE, FLOAT64_TYPE],
                one_qb_row,
            ),
        }
    }

    fn add_to_extension<'e>(
        &self,
        ext: &'e mut Extension,
    ) -> Result<&'e OpDef, ExtensionBuildError> {
        let name = self.name().into();
        let FunctionType { input, output, .. } = self.signature();
        ext.add_op_custom_sig(
            name,
            format!("TKET 2 quantum op: {}", self.name()),
            vec![],
            HashMap::from_iter([(
                "commutation".to_string(),
                serde_yaml::to_value(self.qubit_commutation()).unwrap(),
            )]),
            vec![],
            move |_: &_| Ok(FunctionType::new(input.clone(), output.clone())),
        )
    }

    fn from_extension_name(
        extension: &ExtensionId,
        op_name: &str,
    ) -> Result<Self, Self::LoadError> {
        if extension != &EXTENSION_ID {
            return Err(NotT2Op);
        }
        Self::from_str(op_name).map_err(|_| NotT2Op)
    }
}

impl T2Op {
    pub(crate) fn qubit_commutation(&self) -> Vec<(usize, Pauli)> {
        use T2Op::*;

        match self {
            X | RxF64 => vec![(0, Pauli::X)],
            T | Z | S | Tdg | Sdg | RzF64 | Measure => vec![(0, Pauli::Z)],
            CX => vec![(0, Pauli::Z), (1, Pauli::X)],
            ZZMax | ZZPhase | CZ => vec![(0, Pauli::Z), (1, Pauli::Z)],
            // by default, no commutation
            _ => vec![],
        }
    }

    /// Check if this op is a quantum op.
    pub fn is_quantum(&self) -> bool {
        use T2Op::*;
        match self {
            H | CX | T | S | X | Y | Z | Tdg | Sdg | ZZMax | RzF64 | RxF64 | PhasedX | ZZPhase
            | CZ | TK1 => true,
            AngleAdd | Measure => false,
        }
    }
}

/// Initialize a new custom symbolic expression constant op from a string.
pub fn symbolic_constant_op(s: &str) -> OpType {
    let value: serde_yaml::Value = s.into();
    let l: LeafOp = EXTENSION
        .instantiate_extension_op(
            &SYM_OP_ID,
            vec![TypeArg::Opaque {
                arg: CustomTypeArg::new(SYM_EXPR_T.clone(), value).unwrap(),
            }],
            &REGISTRY,
        )
        .unwrap()
        .into();
    l.into()
}

/// match against a symbolic constant
pub(crate) fn match_symb_const_op(op: &OpType) -> Option<&str> {
    if let OpType::LeafOp(LeafOp::CustomOp(e)) = op {
        match e.as_ref() {
            ExternalOp::Extension(e)
                if e.def().name() == &SYM_OP_ID && e.def().extension() == &EXTENSION_ID =>
            {
                // TODO also check extension name

                let Some(TypeArg::Opaque { arg }) = e.args().get(0) else {
                    panic!("should be an opaque type arg.")
                };

                let serde_yaml::Value::String(s) = &arg.value else {
                    panic!("unexpected yaml value.")
                };

                Some(s)
            }
            ExternalOp::Opaque(_) => todo!(),
            _ => None,
        }
    } else {
        None
    }
}

// From implementations could be made generic over SimpleOpEnum
impl From<T2Op> for LeafOp {
    fn from(op: T2Op) -> Self {
        EXTENSION
            .instantiate_extension_op(op.name(), [], &REGISTRY)
            .unwrap()
            .into()
    }
}

impl From<T2Op> for OpType {
    fn from(op: T2Op) -> Self {
        let l: LeafOp = op.into();
        l.into()
    }
}

impl TryFrom<OpType> for T2Op {
    type Error = NotT2Op;

    fn try_from(op: OpType) -> Result<Self, Self::Error> {
        Self::try_from(&op)
    }
}

impl TryFrom<&OpType> for T2Op {
    type Error = NotT2Op;

    fn try_from(op: &OpType) -> Result<Self, Self::Error> {
        let OpType::LeafOp(leaf) = op else {
            return Err(NotT2Op);
        };
        leaf.try_into()
    }
}

impl TryFrom<&LeafOp> for T2Op {
    type Error = NotT2Op;

    fn try_from(op: &LeafOp) -> Result<Self, Self::Error> {
        match op {
            LeafOp::CustomOp(b) => match b.as_ref() {
                ExternalOp::Extension(e) => Self::try_from_op_def(e.def()),
                ExternalOp::Opaque(o) => from_extension_name(o.extension(), o.name()),
            },
            _ => Err(NotT2Op),
        }
    }
}

impl TryFrom<LeafOp> for T2Op {
    type Error = NotT2Op;

    fn try_from(op: LeafOp) -> Result<Self, Self::Error> {
        Self::try_from(&op)
    }
}

/// load all variants of a `SimpleOpEnum` in to an extension as op defs.
pub(crate) fn load_all_ops<T: SimpleOpEnum>(
    extension: &mut Extension,
) -> Result<(), ExtensionBuildError> {
    for op in T::all_variants() {
        op.add_to_extension(extension)?;
    }
    Ok(())
}
#[cfg(test)]
pub(crate) mod test {

    use std::sync::Arc;

    use hugr::{extension::OpDef, Hugr};
    use rstest::{fixture, rstest};

    use super::T2Op;
    use crate::extension::{TKET2_EXTENSION as EXTENSION, TKET2_EXTENSION_ID as EXTENSION_ID};
    use crate::{circuit::Circuit, ops::SimpleOpEnum, utils::build_simple_circuit};
    fn get_opdef(op: impl SimpleOpEnum) -> Option<&'static Arc<OpDef>> {
        EXTENSION.get_op(op.name())
    }
    #[test]
    fn create_extension() {
        assert_eq!(EXTENSION.name(), &EXTENSION_ID);

        for o in T2Op::all_variants() {
            assert_eq!(T2Op::try_from_op_def(get_opdef(o).unwrap()), Ok(o));
        }
    }

    #[fixture]
    pub(crate) fn t2_bell_circuit() -> Hugr {
        let h = build_simple_circuit(2, |circ| {
            circ.append(T2Op::H, [0])?;
            circ.append(T2Op::CX, [0, 1])?;
            Ok(())
        });

        h.unwrap()
    }

    #[rstest]
    fn check_t2_bell(t2_bell_circuit: Hugr) {
        assert_eq!(t2_bell_circuit.commands().count(), 2);
    }
}
