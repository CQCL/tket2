use std::sync::{Arc, Weak};

use crate::extension::rotation::rotation_type;
use crate::extension::sympy::{SympyOpDef, SYM_OP_ID};
use crate::extension::{TKET2_EXTENSION, TKET2_EXTENSION_ID as EXTENSION_ID};
use hugr::ops::custom::ExtensionOp;
use hugr::ops::NamedOp;
use hugr::types::Type;
use hugr::{
    extension::{
        prelude::{bool_t, option_type, qb_t},
        simple_op::{try_from_name, MakeOpDef, MakeRegisteredOp},
        ExtensionId, OpDef, SignatureFunc,
    },
    ops::OpType,
    type_row,
    types::{type_param::TypeArg, Signature},
};

use derive_more::{Display, Error};
use serde::{Deserialize, Serialize};
use strum::{EnumIter, EnumString, IntoStaticStr};

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
    MeasureFree,
    QAlloc,
    TryQAlloc,
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

#[derive(Display, Debug, Error, PartialEq, Clone)]
#[display("{} is not a Tk2Op.", op.name())]
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
    fn init_signature(&self, _extension_ref: &std::sync::Weak<hugr::Extension>) -> SignatureFunc {
        use Tk2Op::*;
        match self {
            H | T | S | X | Y | Z | Tdg | Sdg | Reset => Signature::new_endo(qb_t()),
            CX | CZ | CY => Signature::new_endo(vec![qb_t(); 2]),
            Toffoli => Signature::new_endo(vec![qb_t(); 3]),
            Measure => Signature::new(qb_t(), vec![qb_t(), bool_t()]),
            MeasureFree => Signature::new(qb_t(), bool_t()),
            Rz | Rx | Ry => Signature::new(vec![qb_t(), rotation_type()], qb_t()),
            CRz => Signature::new(vec![qb_t(), qb_t(), rotation_type()], vec![qb_t(); 2]),
            QAlloc => Signature::new(type_row![], qb_t()),
            TryQAlloc => Signature::new(type_row![], Type::from(option_type(qb_t()))),
            QFree => Signature::new(qb_t(), type_row![]),
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
        try_from_name(op_def.name(), op_def.extension_id())
    }

    fn extension_ref(&self) -> Weak<hugr::Extension> {
        Arc::downgrade(&TKET2_EXTENSION)
    }
}

impl MakeRegisteredOp for Tk2Op {
    fn extension_id(&self) -> ExtensionId {
        EXTENSION_ID.to_owned()
    }

    fn extension_ref(&self) -> Weak<hugr::Extension> {
        Arc::<hugr::Extension>::downgrade(&TKET2_EXTENSION)
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
            Measure | MeasureFree | QAlloc | TryQAlloc | QFree | Reset => false,
        }
    }
}

/// Initialize a new custom symbolic expression constant op from a string.
pub fn symbolic_constant_op(arg: String) -> OpType {
    SympyOpDef.with_expr(arg).into()
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
        if e.def().name() == &SYM_OP_ID && e.def().extension_id() == &EXTENSION_ID {
            Some(symbol_from_typeargs(e.args()))
        } else {
            None
        }
    } else {
        None
    }
}

#[cfg(test)]
pub(crate) mod test {

    use std::str::FromStr;
    use std::sync::Arc;

    use hugr::builder::{DFGBuilder, Dataflow, DataflowHugr};
    use hugr::extension::prelude::{bool_t, option_type, qb_t};
    use hugr::extension::simple_op::MakeOpDef;
    use hugr::extension::{prelude::UnwrapBuilder as _, OpDef};
    use hugr::ops::NamedOp;
    use hugr::types::Signature;
    use hugr::{type_row, CircuitUnit, HugrView};
    use itertools::Itertools;
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
    fn try_qalloc_measure_free() {
        let mut b = DFGBuilder::new(Signature::new(type_row![], bool_t())).unwrap();

        let try_q = b.add_dataflow_op(Tk2Op::TryQAlloc, []).unwrap().out_wire(0);
        let [q] = b.build_unwrap_sum(1, option_type(qb_t()), try_q).unwrap();
        let measured = b
            .add_dataflow_op(Tk2Op::MeasureFree, [q])
            .unwrap()
            .out_wire(0);
        let h = b.finish_hugr_with_outputs([measured]).unwrap();

        let top_ops = h.children(h.root()).map(|n| h.get_optype(n)).collect_vec();

        assert_eq!(top_ops.len(), 5);
        // first two are I/O
        assert_eq!(
            Tk2Op::from_op(top_ops[2].as_extension_op().unwrap()).unwrap(),
            Tk2Op::TryQAlloc
        );
        assert!(top_ops[3].is_conditional());
        assert_eq!(
            Tk2Op::from_op(top_ops[4].as_extension_op().unwrap()).unwrap(),
            Tk2Op::MeasureFree
        );
    }
    #[test]
    fn tk2op_properties() {
        for op in Tk2Op::iter() {
            // The exposed name should start with "tket2.quantum."
            assert!(op.exposed_name().starts_with(&EXTENSION_ID.to_string()));

            let ext_op = op.into_extension_op();
            assert_eq!(ext_op.args(), &[]);
            assert_eq!(ext_op.def().extension_id(), &EXTENSION_ID);
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
