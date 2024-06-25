use crate::extension::{
    SYM_EXPR_T, SYM_OP_ID, TKET2_EXTENSION as EXTENSION, TKET2_EXTENSION_ID as EXTENSION_ID,
};
use hugr::ops::NamedOp;
use hugr::{
    extension::{
        prelude::{BOOL_T, QB_T},
        simple_op::{try_from_name, MakeExtensionOp, MakeOpDef, MakeRegisteredOp},
        ExtensionId, OpDef, SignatureFunc,
    },
    ops::{CustomOp, OpType},
    std_extensions::arithmetic::float_types::FLOAT64_TYPE,
    type_row,
    types::{
        type_param::{CustomTypeArg, TypeArg},
        FunctionType,
    },
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
#[non_exhaustive]
/// Simple enum of tket 2 quantum operations.
//
// When adding new operations, make sure to also edit:
// - `Tk2Op::is_quantum` and `Tk2Op::qubit_commutation` in this same file
// - `tket2/src/serialize/pytket/op/native.rs`: For pytket operation equivalence
// - `tket2-py/tket2/ops.py`: For pure-python operation definitions
pub enum Tk2Op {
    // Single-qubit gates
    /// X rotation
    X,
    /// Y rotation
    Y,
    /// Z rotation
    Z,
    /// Hadamard gate
    H,
    /// T gate
    T,
    /// Inverse T gate
    Tdg,
    /// S gate
    S,
    /// Inverse S gate
    Sdg,
    /// SX gate
    SX,
    /// Inverse SX gate
    SXdg,
    /// V gate
    V,
    /// Inverse V gate
    Vdg,
    /// X rotation with an angle parameter in half-turns
    RxF64,
    /// Y rotation with an angle parameter in half-turns
    RyF64,
    /// Z rotation with an angle parameter in half-turns
    RzF64,
    /// TK1 gate
    ///
    /// TK1(α, β, γ) = Rz(α) Rx(β) Rz(γ)
    TK1,
    /// U1 gate
    ///
    /// U1(α) = U3(0, 0, α)
    ///
    /// See [Tk2Op::U3]
    U1,
    /// U2 gate
    ///
    /// U2(α, β) = U3(1/2, α, β)
    ///
    /// See [Tk2Op::U3]
    U2,
    /// U gate used by IBM
    ///
    /// Parametric on three Euler angles
    ///
    /// See [https://docs.quantum.ibm.com/api/qiskit/qiskit.circuit.library.UGate]
    U3,

    /// Parametric X rotation with two angle parameters
    ///
    /// PhasedX(α, β) = RzF64(β) RxF64(α) RzF64(-β)
    PhasedX,

    // Controlled single-qubit gates
    /// Controlled X gate
    CX,
    /// Controlled Y gate
    CY,
    /// Controlled Z gate
    CZ,
    /// Controlled Hadamard gate
    CH,
    /// Controlled T gate
    CT,
    /// Controlled inverse T gate
    CTdg,
    /// Controlled S gate
    CS,
    /// Controlled inverse S gate
    CSdg,
    /// Controlled SX gate
    CSX,
    /// Controlled inverse SX gate
    CSXdg,
    /// Controlled V gate
    CV,
    /// Controlled inverse V gate
    CVdg,
    /// Controlled X rotation with an angle parameter in half-turns
    CRxF64,
    /// Controlled Y rotation with an angle parameter in half-turns
    CRyF64,
    /// Controlled Z rotation with an angle parameter in half-turns
    CRzF64,
    /// Controlled U1 gate
    ///
    /// See [Tk2Op::U1]
    CU1,
    /// Controlled U2 gate
    ///
    /// See [Tk2Op::U2]
    CU2,
    /// Controlled U3 gate
    ///
    /// See [Tk2Op::U3]
    CU3,
    /// GPi gate
    GPI,
    /// GPi2 gate
    GPI2,

    // Other multi-qubit gates
    /// ZZPhase gate, with a parameter in half-turns
    XXPhase,
    /// ZZPhase gate, with a parameter in half-turns
    YYPhase,
    /// ZZPhase gate, with a parameter in half-turns
    ZZPhase,
    /// Maximally entangling ZZPhase gate
    ZZMax,
    /// TK2 gate
    ///
    /// TK2(α, β, γ) = XXPhase(α) YYPhase(β) ZZPhase(γ)
    TK2,
    /// Swap two qubits
    SWAP,
    /// Controlled qubit swap
    CSWAP,
    /// Three-qubit gate that swaps the first and third qubits
    BRIDGE,
    /// Toffoli gate, or double-controlled X gate
    CCX,
    /// ECR gate
    ECR,
    /// ISWAP gate
    ///
    /// Also known as an XY gate
    ISWAP,
    /// Maximally entangling ISWAP gate
    ///
    /// ISWAPMax = ISWAP(1)
    ISWAPMax,
    /// Phased ISWAP
    PhasedISWAP,
    /// ESWAP gate
    ESWAP,
    /// Three-qubit XX phase gate
    ///
    /// XXPhase3(α)[q0, q1, q2] = XXPhase(α)[q0, q1] XXPhase(α)[q1, q2] XXPhase(α)[q0, q2]
    XXPhase3,
    /// FSim gate
    FSim,
    /// Sycamore gate
    ///
    /// Sycamore = FSim(1/2, 1/6)
    Sycamore,
    /// AAMS gate
    AAMS,

    // Non unitary operations
    /// Measurement
    Measure,
    /// Qubit allocation
    QAlloc,
    /// Qubit deallocation
    QFree,
    /// Reset qubit to |0>
    Reset,

    // Angle manipulation
    /// Add two angles together
    AngleAdd,
    /*
    TODO: Gates with parametric signatures

    Barrier,
    /// N-qubit controlled X gate
    CnX,
    /// N-qubit controlled Y gate
    CnY,
    /// N-qubit controlled Z gate
    CnZ,
    /// N-qubit controlled Ry gate
    CnRy,
    /// N-qubit gate composed of identical PhasedX gates in parallel
    NPhasedX,
    */
}

impl Tk2Op {
    /// Expose the operation names directly in Tk2Op
    pub fn exposed_name(&self) -> smol_str::SmolStr {
        <Tk2Op as Into<OpType>>::into(*self).name()
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

        // Define the type rows once to avoid repeated static allocations.
        let row_1q = type_row![QB_T];
        let row_1q_1param = type_row![QB_T, FLOAT64_TYPE];
        let row_1q_2param = type_row![QB_T, FLOAT64_TYPE, FLOAT64_TYPE];
        let row_1q_3param = type_row![QB_T, FLOAT64_TYPE, FLOAT64_TYPE, FLOAT64_TYPE];
        let row_2q = type_row![QB_T, QB_T];
        let row_2q_1param = type_row![QB_T, QB_T, FLOAT64_TYPE];
        let row_2q_2param = type_row![QB_T, QB_T, FLOAT64_TYPE, FLOAT64_TYPE];
        let row_2q_3param = type_row![QB_T, QB_T, FLOAT64_TYPE, FLOAT64_TYPE, FLOAT64_TYPE];
        let row_3q = type_row![QB_T, QB_T, QB_T];
        let row_3q_1param = type_row![QB_T, QB_T, QB_T, FLOAT64_TYPE];

        match self {
            // 1 qubit gates
            X | Y | Z | H | T | Tdg | S | Sdg | SX | SXdg | V | Vdg | Reset => {
                FunctionType::new_endo(row_1q.clone())
            }
            // 1 qubit, 1 parameter
            RxF64 | RyF64 | RzF64 | U1 | GPI | GPI2 => FunctionType::new(row_1q_1param, row_1q),
            // 1 qubit, 2 parameters
            PhasedX | U2 => FunctionType::new(row_1q_2param, row_1q),
            // 1 qubit, 3 parameters
            TK1 | U3 => FunctionType::new(row_1q_3param, row_1q),
            // 2 qubit gates
            CX | CY | CZ | CH | CT | CTdg | CS | CSdg | CSX | CSXdg | CV | CVdg | ZZMax | SWAP
            | ISWAPMax | ECR | Sycamore => FunctionType::new_endo(row_2q),
            // 2 qubits, 1 parameter
            CRxF64 | CRyF64 | CRzF64 | XXPhase | YYPhase | ZZPhase | CU1 | ISWAP | ESWAP => {
                FunctionType::new(row_2q_1param, row_2q)
            }
            // 2 qubits, 2 parameters
            CU2 | PhasedISWAP | FSim => FunctionType::new(row_2q_2param, row_2q),
            // 2 qubits, 3 parameters
            TK2 | CU3 | AAMS => FunctionType::new(row_2q_3param, row_2q),
            // 3 qubit gates
            CSWAP | BRIDGE | CCX => FunctionType::new_endo(row_3q),
            // 3 qubits, 1 parameter
            XXPhase3 => FunctionType::new(row_3q_1param, row_3q),
            // Bit operations
            Measure => FunctionType::new(row_1q, type_row![QB_T, BOOL_T]),
            QAlloc => FunctionType::new(type_row![], row_1q),
            QFree => FunctionType::new(row_1q, type_row![]),
            // Parameter gates
            AngleAdd => FunctionType::new(
                type_row![FLOAT64_TYPE, FLOAT64_TYPE],
                type_row![FLOAT64_TYPE],
            ),
        }
        .into()
    }

    fn post_opdef(&self, def: &mut OpDef) {
        def.add_misc(
            "commutation",
            serde_yaml::to_value(self.qubit_commutation()).unwrap(),
        );
    }

    fn from_def(op_def: &OpDef) -> Result<Self, hugr::extension::simple_op::OpLoadError> {
        try_from_name(op_def.name())
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

        // TODO: Review missing commutation relations
        match self {
            // 1 qubit X commutation
            X | RxF64 | SX | SXdg => vec![(0, Pauli::X)],
            // 1 qubit Y commutation
            Y | RyF64 => vec![(0, Pauli::Y)],
            // 1 qubit Z commutation
            Z | T | Tdg | S | Sdg | V | Vdg | RzF64 | Measure | Reset => vec![(0, Pauli::Z)],
            // Controlled X commutation
            CX | CRxF64 | CSX | CSXdg => vec![(0, Pauli::Z), (1, Pauli::X)],
            // Controlled Y commutation
            CY | CRyF64 => vec![(0, Pauli::Z), (1, Pauli::Y)],
            // Controlled Z commutation
            CZ | CT | CTdg | CS | CSdg | CV | CVdg | CRzF64 => {
                vec![(0, Pauli::Z), (1, Pauli::Z)]
            }
            // Other controlled ops
            CH | CU1 | CU2 | CU3 => vec![(0, Pauli::Z)],
            // Multi-qubit gates
            CCX => vec![(0, Pauli::Z), (1, Pauli::Z), (2, Pauli::X)],
            // by default, no commutation
            _ => vec![],
        }
    }

    /// Check if this op is a quantum op.
    pub fn is_quantum(&self) -> bool {
        use Tk2Op::*;
        !matches!(self, AngleAdd | Measure | QAlloc | QFree | Reset)
    }
}

/// Initialize a new custom symbolic expression constant op from a string.
pub fn symbolic_constant_op(s: &str) -> OpType {
    let value: serde_yaml::Value = s.into();
    EXTENSION
        .instantiate_extension_op(
            &SYM_OP_ID,
            vec![TypeArg::Opaque {
                arg: CustomTypeArg::new(SYM_EXPR_T.clone(), value).unwrap(),
            }],
            &REGISTRY,
        )
        .unwrap()
        .into()
}

/// match against a symbolic constant
pub(crate) fn match_symb_const_op(op: &OpType) -> Option<String> {
    // Extract the symbol for a symbolic operation node.
    let symbol_from_typeargs = |args: &[TypeArg]| -> String {
        args.first()
            .and_then(|arg| match arg {
                TypeArg::Opaque { arg } => match &arg.value {
                    serde_yaml::Value::String(s) => Some(s.clone()),
                    _ => None,
                },
                _ => None,
            })
            .unwrap_or_else(|| panic!("Found an invalid type arg in a symbolic operation node."))
    };

    if let OpType::CustomOp(custom_op) = op {
        match custom_op {
            CustomOp::Extension(e)
                if e.def().name() == &SYM_OP_ID && e.def().extension() == &EXTENSION_ID =>
            {
                Some(symbol_from_typeargs(e.args()))
            }
            CustomOp::Opaque(e) if e.name() == &SYM_OP_ID && e.extension() == &EXTENSION_ID => {
                Some(symbol_from_typeargs(e.args()))
            }
            _ => None,
        }
    } else {
        None
    }
}

impl TryFrom<&OpType> for Tk2Op {
    type Error = NotTk2Op;

    fn try_from(op: &OpType) -> Result<Self, Self::Error> {
        {
            let OpType::CustomOp(custom_op) = op else {
                return Err(NotTk2Op { op: op.clone() });
            };

            match custom_op {
                CustomOp::Extension(ext) => Tk2Op::from_extension_op(ext).ok(),
                CustomOp::Opaque(opaque) => match opaque.extension() == &EXTENSION_ID {
                    true => try_from_name(opaque.name()).ok(),
                    false => None,
                },
            }
            .ok_or_else(|| NotTk2Op { op: op.clone() })
        }
    }
}

#[cfg(test)]
pub(crate) mod test {

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
}
