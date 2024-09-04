use crate::extension::angle::ANGLE_TYPE;
use crate::extension::{
    SYM_OP_ID, TKET2_EXTENSION as EXTENSION, TKET2_EXTENSION_ID as EXTENSION_ID,
};
use hugr::ops::custom::ExtensionOp;
use hugr::ops::NamedOp;
use hugr::{
    extension::{
        prelude::{BOOL_T, QB_T},
        simple_op::{try_from_name, MakeExtensionOp, MakeOpDef, MakeRegisteredOp},
        ExtensionId, OpDef, SignatureFunc,
    },
    ops::OpType,
    type_row,
    types::{type_param::TypeArg, Signature},
};

use serde::{Deserialize, Serialize};

use strum_macros::{Display, EnumIter, EnumString, IntoStaticStr};
use thiserror::Error;

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
#[allow(missing_docs)]
#[non_exhaustive]
/// Simple enum of tket 2 quantum operations.
pub enum Tk2Op {
    H,
    CX,
    CY,
    CZ,
    CRz,
    T,
    Tdg,
    S,
    Sdg,
    X,
    Y,
    Z,
    Rx,
    Ry,
    Rz,
    Toffoli,
    Measure,
    QAlloc,
    QFree,
    Reset,
}

impl Tk2Op {
    /// Expose the operation names directly in Tk2Op
    pub fn exposed_name(&self) -> smol_str::SmolStr {
        <Tk2Op as Into<OpType>>::into(*self).name()
    }

    /// Wraps the operation in an [`ExtensionOp`]
    pub fn into_extension_op(self) -> ExtensionOp {
        <Self as MakeRegisteredOp>::to_extension_op(self)
            .expect("Failed to convert to extension op.")
    }
}

/// Whether an op is a given Tk2Op.
pub fn op_matches(op: &OpType, tk2op: Tk2Op) -> bool {
    op.name() == tk2op.exposed_name()
}

#[derive(
    Clone, Copy, Debug, Serialize, Deserialize, EnumIter, Display, PartialEq, PartialOrd, EnumString,
)]
#[allow(missing_docs)]
/// Simple enum representation of Pauli matrices.
pub enum Pauli {
    I,
    X,
    Y,
    Z,
}

#[derive(Debug, Error, PartialEq, Clone)]
#[error("{} is not a Tk2Op.", op.name())]
pub struct NotTk2Op {
    /// The offending operation.
    pub op: OpType,
}

impl Pauli {
    /// Check if this pauli commutes with another.
    pub fn commutes_with(&self, other: Self) -> bool {
        *self == Pauli::I || other == Pauli::I || *self == other
    }
}
impl MakeOpDef for Tk2Op {
    fn signature(&self) -> SignatureFunc {
        use Tk2Op::*;
        let one_qb_row = type_row![QB_T];
        match self {
            H | T | S | X | Y | Z | Tdg | Sdg | Reset => Signature::new_endo(one_qb_row),
            CX | CZ | CY => Signature::new_endo(type_row![QB_T; 2]),
            Toffoli => Signature::new_endo(type_row![QB_T; 3]),
            Measure => Signature::new(one_qb_row, type_row![QB_T, BOOL_T]),
            Rz | Rx | Ry => Signature::new(type_row![QB_T, ANGLE_TYPE], one_qb_row),
            CRz => Signature::new(type_row![QB_T, QB_T, ANGLE_TYPE], type_row![QB_T; 2]),
            QAlloc => Signature::new(type_row![], one_qb_row),
            QFree => Signature::new(one_qb_row, type_row![]),
        }
        .into()
    }

    fn extension(&self) -> ExtensionId {
        EXTENSION_ID.to_owned()
    }

    fn post_opdef(&self, def: &mut OpDef) {
        def.add_misc(
            "commutation",
            serde_json::to_value(self.qubit_commutation()).unwrap(),
        );
    }

    fn from_def(op_def: &OpDef) -> Result<Self, hugr::extension::simple_op::OpLoadError> {
        try_from_name(op_def.name(), &EXTENSION_ID)
    }
}

impl MakeRegisteredOp for Tk2Op {
    fn extension_id(&self) -> ExtensionId {
        EXTENSION_ID.to_owned()
    }

    fn registry<'s, 'r: 's>(&'s self) -> &'r hugr::extension::ExtensionRegistry {
        &REGISTRY
    }
}

impl Tk2Op {
    pub(crate) fn qubit_commutation(&self) -> Vec<(usize, Pauli)> {
        use Tk2Op::*;

        match self {
            X | Rx => vec![(0, Pauli::X)],
            Y => vec![(0, Pauli::Y)],
            T | Z | S | Tdg | Sdg | Rz | Measure => vec![(0, Pauli::Z)],
            CX => vec![(0, Pauli::Z), (1, Pauli::X)],
            CZ => vec![(0, Pauli::Z), (1, Pauli::Z)],
            // by default, no commutation
            _ => vec![],
        }
    }

    /// Check if this op is a quantum op.
    pub fn is_quantum(&self) -> bool {
        use Tk2Op::*;
        match self {
            H | CX | T | S | X | Y | Z | Tdg | Sdg | Rz | Rx | Toffoli | Ry | CZ | CY | CRz => true,
            Measure | QAlloc | QFree | Reset => false,
        }
    }
}

/// Initialize a new custom symbolic expression constant op from a string.
pub fn symbolic_constant_op(arg: String) -> OpType {
    EXTENSION
        .instantiate_extension_op(&SYM_OP_ID, vec![arg.into()], &REGISTRY)
        .unwrap()
        .into()
}

/// match against a symbolic constant
pub(crate) fn match_symb_const_op(op: &OpType) -> Option<String> {
    // Extract the symbol for a symbolic operation node.
    let symbol_from_typeargs = |args: &[TypeArg]| -> String {
        args.first()
            .and_then(|arg| match arg {
                TypeArg::String { arg } => Some(arg.clone()),
                _ => None,
            })
            .unwrap_or_else(|| panic!("Found an invalid type arg in a symbolic operation node."))
    };

    if let OpType::ExtensionOp(e) = op {
        if e.def().name() == &SYM_OP_ID && e.def().extension() == &EXTENSION_ID {
            Some(symbol_from_typeargs(e.args()))
        } else {
            None
        }
    } else {
        None
    }
}

impl TryFrom<&OpType> for Tk2Op {
    type Error = NotTk2Op;

    fn try_from(op: &OpType) -> Result<Self, Self::Error> {
        {
            match op {
                OpType::ExtensionOp(ext) => Tk2Op::from_extension_op(ext).ok(),
                OpType::OpaqueOp(opaque) => try_from_name(&opaque.name(), &EXTENSION_ID).ok(),
                _ => None,
            }
            .ok_or_else(|| NotTk2Op { op: op.clone() })
        }
    }
}

#[cfg(test)]
pub(crate) mod test {

    use std::str::FromStr;
    use std::sync::Arc;

    use hugr::extension::simple_op::MakeOpDef;
    use hugr::extension::OpDef;
    use hugr::ops::NamedOp;
    use hugr::CircuitUnit;
    use rstest::{fixture, rstest};
    use strum::IntoEnumIterator;

    use super::Tk2Op;
    use crate::circuit::Circuit;
    use crate::extension::{TKET2_EXTENSION as EXTENSION, TKET2_EXTENSION_ID as EXTENSION_ID};
    use crate::utils::build_simple_circuit;
    use crate::Pauli;
    fn get_opdef(op: impl NamedOp) -> Option<&'static Arc<OpDef>> {
        EXTENSION.get_op(&op.name())
    }
    #[test]
    fn create_extension() {
        assert_eq!(EXTENSION.name(), &EXTENSION_ID);

        for o in Tk2Op::iter() {
            assert_eq!(Tk2Op::from_def(get_opdef(o).unwrap()), Ok(o));
        }
    }

    #[fixture]
    pub(crate) fn t2_bell_circuit() -> Circuit {
        let h = build_simple_circuit(2, |circ| {
            circ.append(Tk2Op::H, [0])?;
            circ.append(Tk2Op::CX, [0, 1])?;
            Ok(())
        });

        h.unwrap()
    }

    #[rstest]
    fn check_t2_bell(t2_bell_circuit: Circuit) {
        assert_eq!(t2_bell_circuit.commands().count(), 2);
    }

    #[test]
    fn ancilla_circ() {
        let h = build_simple_circuit(1, |circ| {
            let empty: [CircuitUnit; 0] = []; // requires type annotation
            let ancilla = circ.append_with_outputs(Tk2Op::QAlloc, empty)?[0];
            let ancilla = circ.append_with_outputs(Tk2Op::Reset, [ancilla])?[0];

            let ancilla = circ.append_with_outputs(
                Tk2Op::CX,
                [CircuitUnit::Linear(0), CircuitUnit::Wire(ancilla)],
            )?[0];
            let ancilla = circ.append_with_outputs(Tk2Op::Measure, [ancilla])?[0];
            circ.append_and_consume(Tk2Op::QFree, [ancilla])?;

            Ok(())
        })
        .unwrap();

        // 5 commands: alloc, reset, cx, measure, free
        assert_eq!(h.commands().count(), 5);
    }

    #[test]
    fn tk2op_properties() {
        for op in Tk2Op::iter() {
            // The exposed name should start with "quantum.tket2."
            assert!(op.exposed_name().starts_with(&EXTENSION_ID.to_string()));

            let ext_op = op.into_extension_op();
            assert_eq!(ext_op.args(), &[]);
            assert_eq!(ext_op.def().extension(), &EXTENSION_ID);
            let name = ext_op.def().name();
            assert_eq!(Tk2Op::from_str(name), Ok(op));
        }

        // Other calls
        assert!(Tk2Op::H.is_quantum());
        assert!(!Tk2Op::Measure.is_quantum());

        for (op, pauli) in [
            (Tk2Op::X, Pauli::X),
            (Tk2Op::Y, Pauli::Y),
            (Tk2Op::Z, Pauli::Z),
        ]
        .iter()
        {
            assert_eq!(op.qubit_commutation(), &[(0, *pauli)]);
        }
    }
}
