//! Operations that have corresponding representations in both `pytket` and `tket2`.

use std::borrow::Cow;

use hugr::extension::prelude::{bool_t, qb_t, Noop};

use hugr::ops::{OpTrait, OpType};
use hugr::std_extensions::arithmetic::float_types::float64_type;
use hugr::types::Signature;

use hugr::IncomingPort;
use tket_json_rs::circuit_json;
use tket_json_rs::optype::OpType as Tk1OpType;

use crate::extension::rotation::rotation_type;
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
            Tk2Op::H => Tk1OpType::H,
            Tk2Op::CX => Tk1OpType::CX,
            Tk2Op::CY => Tk1OpType::CY,
            Tk2Op::CZ => Tk1OpType::CZ,
            Tk2Op::CRz => Tk1OpType::CRz,
            Tk2Op::T => Tk1OpType::T,
            Tk2Op::Tdg => Tk1OpType::Tdg,
            Tk2Op::S => Tk1OpType::S,
            Tk2Op::Sdg => Tk1OpType::Sdg,
            Tk2Op::X => Tk1OpType::X,
            Tk2Op::Y => Tk1OpType::Y,
            Tk2Op::Z => Tk1OpType::Z,
            Tk2Op::Rx => Tk1OpType::Rx,
            Tk2Op::Rz => Tk1OpType::Rz,
            Tk2Op::Ry => Tk1OpType::Ry,
            Tk2Op::Toffoli => Tk1OpType::CCX,
            Tk2Op::Reset => Tk1OpType::Reset,
            Tk2Op::Measure => Tk1OpType::Measure,
            // These operations do not have a direct pytket counterpart.
            Tk2Op::MeasureFree => return None,
            Tk2Op::QAlloc | Tk2Op::QFree | Tk2Op::TryQAlloc => {
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
            Tk1OpType::H => Tk2Op::H.into(),
            Tk1OpType::CX => Tk2Op::CX.into(),
            Tk1OpType::CY => Tk2Op::CY.into(),
            Tk1OpType::CZ => Tk2Op::CZ.into(),
            Tk1OpType::CRz => Tk2Op::CRz.into(),
            Tk1OpType::T => Tk2Op::T.into(),
            Tk1OpType::Tdg => Tk2Op::Tdg.into(),
            Tk1OpType::S => Tk2Op::S.into(),
            Tk1OpType::Sdg => Tk2Op::Sdg.into(),
            Tk1OpType::X => Tk2Op::X.into(),
            Tk1OpType::Y => Tk2Op::Y.into(),
            Tk1OpType::Z => Tk2Op::Z.into(),
            Tk1OpType::Rx => Tk2Op::Rx.into(),
            Tk1OpType::Ry => Tk2Op::Ry.into(),
            Tk1OpType::Rz => Tk2Op::Rz.into(),
            Tk1OpType::CCX => Tk2Op::Toffoli.into(),
            Tk1OpType::Reset => Tk2Op::Reset.into(),
            Tk1OpType::Measure => Tk2Op::Measure.into(),
            Tk1OpType::noop => Noop::new(qb_t()).into(),
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

        let mut op = circuit_json::Operation::default();
        op.op_type = serial_op;
        op.n_qb = Some(num_qubits as u32);
        op.params = params;
        op.signature = Some([vec!["Q".into(); num_qubits], vec!["B".into(); num_bits]].concat());
        Some(op)
    }

    /// Returns the dataflow signature for this operation.
    pub fn signature(&self) -> Option<Cow<'_, Signature>> {
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
                .filter(|(_, ty)| [rotation_type(), float64_type()].contains(ty))
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
        let Some(sig) = self.signature().map(Cow::into_owned) else {
            return;
        };
        for ty in sig.input_types() {
            if ty == &qb_t() {
                self.input_qubits += 1;
            } else if ty == &bool_t() {
                self.input_bits += 1;
            } else if [rotation_type(), float64_type()].contains(ty) {
                self.num_params += 1;
            }
        }
        for ty in sig.output_types() {
            if ty == &qb_t() {
                self.output_qubits += 1;
            } else if ty == &bool_t() {
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
