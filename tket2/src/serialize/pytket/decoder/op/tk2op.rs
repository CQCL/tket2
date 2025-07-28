//! Encoder and decoder for tket2 operations with native pytket counterparts.

use std::borrow::Cow;

use hugr::extension::prelude::{bool_t, qb_t, Noop};

use hugr::ops::{OpTrait, OpType};
use hugr::std_extensions::arithmetic::float_types::float64_type;
use hugr::types::Signature;

use hugr::IncomingPort;
use itertools::Itertools;
use tket_json_rs::optype::OpType as Tk1OpType;

use crate::extension::rotation::rotation_type;
use crate::Tk2Op;

/// An operation with a native TKET2 counterpart.
///
/// Note that the signature of the native and serialised operations may differ.
#[derive(Clone, Debug, PartialEq, Default)]
pub(crate) struct NativeOp {
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

    /// Returns the translated tket2 optype for this operation, if it exists.
    pub fn try_from_serial_optype(
        serial_op: Tk1OpType,
        num_qubits: usize,
        num_bits: usize,
    ) -> Option<Self> {
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
            Tk1OpType::V => Tk2Op::V.into(),
            Tk1OpType::Vdg => Tk2Op::Vdg.into(),
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
            Tk1OpType::Barrier => {
                let qbs = std::iter::repeat_n(qb_t(), num_qubits);
                let bs = std::iter::repeat_n(bool_t(), num_bits);
                let types = qbs.chain(bs).collect_vec();
                hugr::extension::prelude::Barrier::new(types).into()
            }
            _ => {
                return None;
            }
        };
        Some(Self::new(op, Some(serial_op)))
    }

    /// Returns the dataflow signature for this operation.
    pub fn signature(&self) -> Option<Cow<'_, Signature>> {
        self.op.dataflow_signature()
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
