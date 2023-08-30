use std::collections::HashMap;

use hugr::{
    extension::{
        prelude::{BOOL_T, QB_T},
        ExtensionBuildError, ExtensionId, OpDef,
    },
    ops::{custom::ExternalOp, LeafOp, OpType},
    std_extensions::arithmetic::float_types::FLOAT64_TYPE,
    type_row,
    types::FunctionType,
    Extension,
};
use lazy_static::lazy_static;
use serde::{Deserialize, Serialize};
use std::str::FromStr;
use strum::IntoEnumIterator;
use strum_macros::{Display, EnumIter, EnumString, IntoStaticStr};

/// Name of tket 2 extension.
pub const EXTENSION_ID: ExtensionId = ExtensionId::new_inline("quantum.tket2");

#[derive(
    Clone, Copy, Debug, Serialize, Deserialize, EnumIter, IntoStaticStr, EnumString, PartialEq,
)]
#[allow(missing_docs)]
/// Simple enum of tket 2 quantum operations.
pub enum T2Op {
    H,
    CX,
    T,
    S,
    X,
    Y,
    Z,
    Tadj,
    Sadj,
    ZZMax,
    Measure,
    RzF64,
}
#[derive(Clone, Copy, Debug, Serialize, Deserialize, EnumIter, Display, PartialEq, PartialOrd)]
#[allow(missing_docs)]
/// Simple enum representation of Pauli matrices.
pub enum Pauli {
    I,
    X,
    Y,
    Z,
}

// this trait could be implemented in Hugr
trait SimpleOpEnum: Into<&'static str> + FromStr + Copy + IntoEnumIterator {
    fn signature(&self) -> FunctionType;
    fn name(&self) -> &str {
        (*self).into()
    }
    fn try_from_op_def(op_def: &OpDef) -> Result<Self, <Self as FromStr>::Err> {
        Self::from_str(op_def.name())
    }
    fn add_to_extension<'e>(
        &self,
        ext: &'e mut Extension,
    ) -> Result<&'e OpDef, ExtensionBuildError>;

    fn all_variants() -> <Self as IntoEnumIterator>::Iterator {
        <Self as IntoEnumIterator>::iter()
    }
}

impl Pauli {
    /// Check if this pauli commutes with another.
    pub fn commutes_with(&self, other: Self) -> bool {
        *self == Pauli::I || other == Pauli::I || *self == other
    }
}
impl SimpleOpEnum for T2Op {
    fn signature(&self) -> FunctionType {
        use T2Op::*;
        let one_qb_row = type_row![QB_T];
        let two_qb_row = type_row![QB_T, QB_T];
        match self {
            H | T | S | X | Y | Z | Tadj | Sadj => {
                FunctionType::new(one_qb_row.clone(), one_qb_row)
            }
            CX | ZZMax => FunctionType::new(two_qb_row.clone(), two_qb_row),
            Measure => FunctionType::new(one_qb_row, type_row![QB_T, BOOL_T]),
            RzF64 => FunctionType::new(type_row![QB_T, FLOAT64_TYPE], one_qb_row),
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
}

impl T2Op {
    pub(crate) fn qubit_commutation(&self) -> Vec<(usize, Pauli)> {
        use T2Op::*;

        match self {
            X => vec![(0, Pauli::X)],
            T | Z | S | Tadj | Sadj | RzF64 => vec![(0, Pauli::Z)],
            CX => vec![(0, Pauli::Z), (1, Pauli::X)],
            ZZMax => vec![(0, Pauli::Z), (1, Pauli::Z)],
            // by default, no commutation
            _ => vec![],
        }
    }
}

fn extension() -> Extension {
    let mut e = Extension::new(EXTENSION_ID);
    load_all_ops::<T2Op>(&mut e).expect("add fail");

    e
}

lazy_static! {
    static ref EXTENSION: Extension = extension();
}

// From implementations could be made generic over SimpleOpEnum
impl From<T2Op> for LeafOp {
    fn from(op: T2Op) -> Self {
        EXTENSION
            .instantiate_extension_op(op.name(), [])
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
    type Error = &'static str;

    fn try_from(op: OpType) -> Result<Self, Self::Error> {
        let leaf: LeafOp = op.try_into().map_err(|_| "not a leaf.")?;
        match leaf {
            LeafOp::CustomOp(b) => match *b {
                ExternalOp::Extension(e) => {
                    Self::try_from_op_def(e.def()).map_err(|_| "not a T2Op")
                }
                ExternalOp::Opaque(_) => todo!(),
            },
            _ => Err("not a custom."),
        }
    }
}
fn load_all_ops<T: SimpleOpEnum>(extension: &mut Extension) -> Result<(), ExtensionBuildError> {
    for op in T::all_variants() {
        op.add_to_extension(extension)?;
    }
    Ok(())
}
#[cfg(test)]
pub(crate) mod test {

    use std::sync::Arc;

    use hugr::{
        builder::{BuildError, CircuitBuilder, DFGBuilder, Dataflow, DataflowHugr},
        extension::{prelude::QB_T, OpDef},
        types::FunctionType,
        Hugr,
    };
    use rstest::fixture;

    use crate::ops::SimpleOpEnum;

    use super::{T2Op, EXTENSION, EXTENSION_ID};
    fn get_opdef(op: impl SimpleOpEnum) -> Option<&'static Arc<OpDef>> {
        EXTENSION.get_op(op.name())
    }
    #[test]
    fn create_extension() {
        assert_eq!(EXTENSION.name(), &EXTENSION_ID);

        for o in T2Op::all_variants() {
            assert_eq!(T2Op::try_from_op_def(get_opdef(o.clone()).unwrap()), Ok(o));
        }
    }

    pub(crate) fn build_simple_circuit(
        num_qubits: usize,
        f: impl FnOnce(&mut CircuitBuilder<DFGBuilder<Hugr>>) -> Result<(), BuildError>,
    ) -> Result<Hugr, BuildError> {
        let qb_row = vec![QB_T; num_qubits];
        let mut h = DFGBuilder::new(FunctionType::new(qb_row.clone(), qb_row))?;

        let qbs = h.input_wires();

        let mut circ = h.as_circuit(qbs.into_iter().collect());

        f(&mut circ)?;

        let qbs = circ.finish();
        h.finish_hugr_with_outputs(qbs)
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
}
