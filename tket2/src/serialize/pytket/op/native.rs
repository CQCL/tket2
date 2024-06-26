//! Operations that have corresponding representations in both `pytket` and `tket2`.

use hugr::extension::prelude::{BOOL_T, QB_T};

use hugr::ops::{Noop, OpTrait, OpType};
use hugr::std_extensions::arithmetic::float_types::FLOAT64_TYPE;
use hugr::types::FunctionType;

use hugr::IncomingPort;
use tket_json_rs::circuit_json;
use tket_json_rs::optype::OpType as Tk1OpType;

use crate::Tk2Op;

/// An operation with a native TKET2 counterpart.
///
/// Note that the signature of the native and serialised operations may differ.
#[derive(Clone, Debug, PartialEq, Default)]
pub struct NativeOp {
    /// The tket2 optype.
    op: OpType,
    /// The corresponding serialised optype.
    ///
    /// Some specific operations do not have a direct pytket counterpart, and must be handled
    /// separately.
    serial_op: Option<Tk1OpType>,
    /// Number of input qubits to the operation.
    pub input_qubits: usize,
    /// Number of output qubits to the operation.
    pub input_bits: usize,
    /// Number of parameters to the operation.
    pub num_params: usize,
    /// Number of output qubits to the operation.
    pub output_qubits: usize,
    /// Number of output bits to the operation.
    pub output_bits: usize,
}

impl NativeOp {
    /// Initialise a new `NativeOp`.
    fn new(op: OpType, serial_op: Option<Tk1OpType>) -> Self {
        let mut native_op = Self {
            op,
            serial_op,
            ..Default::default()
        };
        native_op.compute_counts();
        native_op
    }

    /// Create a new `NativeOp` from a `circuit_json::Operation`.
    pub fn try_from_tk2op(tk2op: Tk2Op) -> Option<Self> {
        let serial_op = match tk2op {
            Tk2Op::X => Tk1OpType::X,
            Tk2Op::Y => Tk1OpType::Y,
            Tk2Op::Z => Tk1OpType::Z,
            Tk2Op::H => Tk1OpType::H,
            Tk2Op::T => Tk1OpType::T,
            Tk2Op::Tdg => Tk1OpType::Tdg,
            Tk2Op::S => Tk1OpType::S,
            Tk2Op::Sdg => Tk1OpType::Sdg,
            Tk2Op::SX => Tk1OpType::SX,
            Tk2Op::SXdg => Tk1OpType::SXdg,
            Tk2Op::V => Tk1OpType::V,
            Tk2Op::Vdg => Tk1OpType::Vdg,
            Tk2Op::RxF64 => Tk1OpType::Rx,
            Tk2Op::RyF64 => Tk1OpType::Ry,
            Tk2Op::RzF64 => Tk1OpType::Rz,
            Tk2Op::TK1 => Tk1OpType::TK1,
            Tk2Op::U1 => Tk1OpType::U1,
            Tk2Op::U2 => Tk1OpType::U2,
            Tk2Op::U3 => Tk1OpType::U3,
            Tk2Op::PhasedX => Tk1OpType::PhasedX,
            Tk2Op::CX => Tk1OpType::CX,
            Tk2Op::CY => Tk1OpType::CY,
            Tk2Op::CZ => Tk1OpType::CZ,
            Tk2Op::CH => Tk1OpType::CH,
            Tk2Op::CT => {
                return None;
            }
            Tk2Op::CTdg => {
                return None;
            }
            Tk2Op::CS => Tk1OpType::CS,
            Tk2Op::CSdg => Tk1OpType::CSdg,
            Tk2Op::CSX => Tk1OpType::CSX,
            Tk2Op::CSXdg => Tk1OpType::CSXdg,
            Tk2Op::CV => Tk1OpType::CV,
            Tk2Op::CVdg => Tk1OpType::CVdg,
            Tk2Op::CRxF64 => Tk1OpType::CRx,
            Tk2Op::CRyF64 => Tk1OpType::CRy,
            Tk2Op::CRzF64 => Tk1OpType::CRz,
            Tk2Op::CU1 => Tk1OpType::CU1,
            Tk2Op::CU2 => {
                return None;
            }
            Tk2Op::CU3 => Tk1OpType::CU3,
            Tk2Op::GPI => Tk1OpType::GPI,
            Tk2Op::GPI2 => Tk1OpType::GPI2,
            Tk2Op::XXPhase => Tk1OpType::XXPhase,
            Tk2Op::YYPhase => Tk1OpType::YYPhase,
            Tk2Op::ZZPhase => Tk1OpType::ZZPhase,
            Tk2Op::ZZMax => Tk1OpType::ZZMax,
            Tk2Op::TK2 => Tk1OpType::TK2,
            Tk2Op::SWAP => Tk1OpType::SWAP,
            Tk2Op::CSWAP => Tk1OpType::CSWAP,
            Tk2Op::BRIDGE => Tk1OpType::BRIDGE,
            Tk2Op::CCX => Tk1OpType::CCX,
            Tk2Op::ECR => Tk1OpType::ECR,
            Tk2Op::ISWAP => Tk1OpType::ISWAP,
            Tk2Op::ISWAPMax => Tk1OpType::ISWAPMax,
            Tk2Op::PhasedISWAP => Tk1OpType::PhasedISWAP,
            Tk2Op::ESWAP => Tk1OpType::ESWAP,
            Tk2Op::XXPhase3 => Tk1OpType::XXPhase3,
            Tk2Op::FSim => Tk1OpType::FSim,
            Tk2Op::Sycamore => Tk1OpType::Sycamore,
            Tk2Op::AAMS => Tk1OpType::AAMS,
            // Non unitary operations.
            Tk2Op::Reset => Tk1OpType::Reset,
            Tk2Op::Measure => Tk1OpType::Measure,
            Tk2Op::AngleAdd => {
                // These operations should be folded into constant before serialisation,
                // or replaced by pytket logic expressions.
                return Some(Self::new(tk2op.into(), None));
            }
            // These operations do not have a direct pytket counterpart.
            Tk2Op::QAlloc | Tk2Op::QFree => {
                // These operations are implicitly supported by the encoding,
                // they do not create an explicit pytket operation but instead
                // add new qubits to the circuit input/output.
                return Some(Self::new(tk2op.into(), None));
            }
        };

        Some(Self::new(tk2op.into(), Some(serial_op)))
    }

    /// Returns the translated tket2 optype for this operation, if it exists.
    pub fn try_from_serial_optype(serial_op: Tk1OpType) -> Option<Self> {
        let op = match serial_op {
            Tk1OpType::X => Tk2Op::X.into(),
            Tk1OpType::Y => Tk2Op::Y.into(),
            Tk1OpType::Z => Tk2Op::Z.into(),
            Tk1OpType::H => Tk2Op::H.into(),
            Tk1OpType::T => Tk2Op::T.into(),
            Tk1OpType::Tdg => Tk2Op::Tdg.into(),
            Tk1OpType::S => Tk2Op::S.into(),
            Tk1OpType::Sdg => Tk2Op::Sdg.into(),
            Tk1OpType::SX => Tk2Op::SX.into(),
            Tk1OpType::SXdg => Tk2Op::SXdg.into(),
            Tk1OpType::V => Tk2Op::V.into(),
            Tk1OpType::Vdg => Tk2Op::Vdg.into(),
            Tk1OpType::Rx => Tk2Op::RxF64.into(),
            Tk1OpType::Ry => Tk2Op::RyF64.into(),
            Tk1OpType::Rz => Tk2Op::RzF64.into(),
            Tk1OpType::TK1 => Tk2Op::TK1.into(),
            Tk1OpType::U1 => Tk2Op::U1.into(),
            Tk1OpType::U2 => Tk2Op::U2.into(),
            Tk1OpType::U3 => Tk2Op::U3.into(),
            Tk1OpType::PhasedX => Tk2Op::PhasedX.into(),
            Tk1OpType::CX => Tk2Op::CX.into(),
            Tk1OpType::CY => Tk2Op::CY.into(),
            Tk1OpType::CZ => Tk2Op::CZ.into(),
            Tk1OpType::CH => Tk2Op::CH.into(),
            Tk1OpType::CS => Tk2Op::CS.into(),
            Tk1OpType::CSdg => Tk2Op::CSdg.into(),
            Tk1OpType::CSX => Tk2Op::CSX.into(),
            Tk1OpType::CSXdg => Tk2Op::CSXdg.into(),
            Tk1OpType::CV => Tk2Op::CV.into(),
            Tk1OpType::CVdg => Tk2Op::CVdg.into(),
            Tk1OpType::CRx => Tk2Op::CRxF64.into(),
            Tk1OpType::CRy => Tk2Op::CRyF64.into(),
            Tk1OpType::CRz => Tk2Op::CRzF64.into(),
            Tk1OpType::CU1 => Tk2Op::CU1.into(),
            Tk1OpType::CU3 => Tk2Op::CU3.into(),
            Tk1OpType::GPI => Tk2Op::GPI.into(),
            Tk1OpType::GPI2 => Tk2Op::GPI2.into(),
            Tk1OpType::XXPhase => Tk2Op::XXPhase.into(),
            Tk1OpType::YYPhase => Tk2Op::YYPhase.into(),
            Tk1OpType::ZZPhase => Tk2Op::ZZPhase.into(),
            Tk1OpType::ZZMax => Tk2Op::ZZMax.into(),
            Tk1OpType::TK2 => Tk2Op::TK2.into(),
            Tk1OpType::SWAP => Tk2Op::SWAP.into(),
            Tk1OpType::CSWAP => Tk2Op::CSWAP.into(),
            Tk1OpType::BRIDGE => Tk2Op::BRIDGE.into(),
            Tk1OpType::CCX => Tk2Op::CCX.into(),
            Tk1OpType::ECR => Tk2Op::ECR.into(),
            Tk1OpType::ISWAP => Tk2Op::ISWAP.into(),
            Tk1OpType::ISWAPMax => Tk2Op::ISWAPMax.into(),
            Tk1OpType::PhasedISWAP => Tk2Op::PhasedISWAP.into(),
            Tk1OpType::ESWAP => Tk2Op::ESWAP.into(),
            Tk1OpType::XXPhase3 => Tk2Op::XXPhase3.into(),
            Tk1OpType::FSim => Tk2Op::FSim.into(),
            Tk1OpType::Sycamore => Tk2Op::Sycamore.into(),
            Tk1OpType::AAMS => Tk2Op::AAMS.into(),

            // Non unitary operations.
            Tk1OpType::Reset => Tk2Op::Reset.into(),
            Tk1OpType::Measure => Tk2Op::Measure.into(),
            Tk1OpType::noop => Noop::new(QB_T).into(),
            _ => {
                return None;
            }
        };
        Some(Self::new(op, Some(serial_op)))
    }

    /// Converts this `NativeOp` into a tket_json_rs operation.
    pub fn serialised_op(&self) -> Option<circuit_json::Operation> {
        let serial_op = self.serial_op.clone()?;

        // Since pytket operations are always linear,
        // use the maximum of input and output bits/qubits.
        let num_qubits = self.input_qubits.max(self.output_qubits);
        let num_bits = self.input_bits.max(self.output_bits);
        let num_params = self.num_params;

        let params = (num_params > 0).then(|| vec!["".into(); num_params]);

        Some(circuit_json::Operation {
            op_type: serial_op,
            n_qb: Some(num_qubits as u32),
            params,
            op_box: None,
            signature: Some([vec!["Q".into(); num_qubits], vec!["B".into(); num_bits]].concat()),
            conditional: None,
        })
    }

    /// Returns the dataflow signature for this operation.
    pub fn signature(&self) -> Option<FunctionType> {
        self.op.dataflow_signature()
    }

    /// Returns the serial optype for this operation.
    ///
    /// Some special operations do not have a direct serialised counterpart, and
    /// should be skipped during serialisation.
    pub fn serial_op(&self) -> Option<&Tk1OpType> {
        self.serial_op.as_ref()
    }

    /// Returns the tket2 optype for this operation.
    pub fn optype(&self) -> &OpType {
        &self.op
    }

    /// Consumes the `NativeOp` and returns the underlying `OpType`.
    pub fn into_optype(self) -> OpType {
        self.op
    }

    /// Returns the ports corresponding to parameters for this operation.
    pub fn param_ports(&self) -> impl Iterator<Item = IncomingPort> + '_ {
        self.signature().into_iter().flat_map(|sig| {
            let types = sig.input_types().to_owned();
            sig.input_ports()
                .zip(types)
                .filter(|(_, ty)| ty == &FLOAT64_TYPE)
                .map(|(port, _)| port)
        })
    }

    /// Update the internal bit/qubit/parameter counts.
    fn compute_counts(&mut self) {
        self.input_bits = 0;
        self.input_qubits = 0;
        self.num_params = 0;
        self.output_bits = 0;
        self.output_qubits = 0;
        let Some(sig) = self.signature() else {
            return;
        };
        for ty in sig.input_types() {
            if ty == &QB_T {
                self.input_qubits += 1;
            } else if ty == &BOOL_T {
                self.input_bits += 1;
            } else if ty == &FLOAT64_TYPE {
                self.num_params += 1;
            }
        }
        for ty in sig.output_types() {
            if ty == &QB_T {
                self.output_qubits += 1;
            } else if ty == &BOOL_T {
                self.output_bits += 1;
            }
        }
    }
}

#[cfg(test)]
mod cfg {
    use super::*;
    use hugr::ops::NamedOp;
    use rstest::rstest;
    use strum::IntoEnumIterator;

    #[rstest]
    fn tk2_optype_correspondence() {
        for tk2op in Tk2Op::iter() {
            let Some(native_op) = NativeOp::try_from_tk2op(tk2op) else {
                // Ignore unsupported ops.
                continue;
            };

            let Some(serial_op) = native_op.serial_op.clone() else {
                // Ignore ops that do not have a serialised equivalent.
                // (But are still handled by the encoder).
                continue;
            };

            let Some(native_op2) = NativeOp::try_from_serial_optype(serial_op.clone()) else {
                panic!(
                    "{} serialises into {serial_op:?}, but failed to be deserialised.",
                    tk2op.name()
                )
            };

            assert_eq!(native_op, native_op2);
        }
    }
}
