//! Encoder and decoder for tket operations with native pytket counterparts.

use std::borrow::Cow;

use hugr::extension::prelude::{bool_t, qb_t, Noop};

use hugr::ops::{OpTrait, OpType};
use hugr::std_extensions::arithmetic::float_types::float64_type;
use hugr::types::Signature;

use hugr::IncomingPort;
use itertools::Itertools;
use tket_json_rs::optype::OpType as Tk1OpType;

use crate::extension::rotation::rotation_type;
use crate::TketOp;

/// An operation with a native TKET counterpart.
///
/// Note that the signature of the native and serialised operations may differ.
#[derive(Clone, Debug, PartialEq, Default)]
pub(crate) struct NativeOp {
    /// The tket optype.
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

    /// Returns the translated tket optype for this operation, if it exists.
    pub fn try_from_serial_optype(
        serial_op: Tk1OpType,
        num_qubits: usize,
        num_bits: usize,
    ) -> Option<Self> {
        let op = match serial_op {
            Tk1OpType::H => TketOp::H.into(),
            Tk1OpType::CX => TketOp::CX.into(),
            Tk1OpType::CY => TketOp::CY.into(),
            Tk1OpType::CZ => TketOp::CZ.into(),
            Tk1OpType::CRz => TketOp::CRz.into(),
            Tk1OpType::T => TketOp::T.into(),
            Tk1OpType::Tdg => TketOp::Tdg.into(),
            Tk1OpType::S => TketOp::S.into(),
            Tk1OpType::Sdg => TketOp::Sdg.into(),
            Tk1OpType::V => TketOp::V.into(),
            Tk1OpType::Vdg => TketOp::Vdg.into(),
            Tk1OpType::X => TketOp::X.into(),
            Tk1OpType::Y => TketOp::Y.into(),
            Tk1OpType::Z => TketOp::Z.into(),
            Tk1OpType::Rx => TketOp::Rx.into(),
            Tk1OpType::Ry => TketOp::Ry.into(),
            Tk1OpType::Rz => TketOp::Rz.into(),
            Tk1OpType::CCX => TketOp::Toffoli.into(),
            Tk1OpType::Reset => TketOp::Reset.into(),
            Tk1OpType::Measure => TketOp::Measure.into(),
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

    /// Returns the tket optype for this operation.
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
