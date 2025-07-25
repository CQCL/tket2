//! This module defines the internal [`Tk1Op`] struct wrapping the logic for
//! going between `tket_json_rs::optype::OpType` and `hugr::ops::OpType`.
//!
//! The `Tk1Op` tries to homogenize the
//! `tket_json_rs::circuit_json::Operation`s coming from the encoded TKET1
//! circuits by ensuring they always define a signature, and computing the
//! explicit count of qubits and linear bits.

use hugr::ops::OpType;
use hugr::IncomingPort;
use tket_json_rs::circuit_json;
use tket_op::NativeOp;

use super::super::extension::OpaqueTk1Op;

pub mod tket_op;

/// An intermediary artifact when converting between TKET1 and TKET operations.
///
/// This enum represents either operations that can be represented natively in TKET,
/// or operations that must be serialised as opaque TKET1 operations.
#[derive(Clone, Debug, PartialEq, derive_more::From)]
pub(crate) enum Tk1Op {
    /// An operation with a native TKET counterpart.
    Native(NativeOp),
    /// An operation without a native TKET counterpart.
    Opaque(OpaqueTk1Op),
}

impl Tk1Op {
    /// Create a new `Tk1Op` from a tket1 `circuit_json::Operation`.
    ///
    /// If `serial_op` defines a signature then `num_qubits` and `num_qubits` are ignored. Otherwise, a signature is synthesised from those parameters.
    pub fn from_serialised_op(
        serial_op: circuit_json::Operation,
        num_qubits: usize,
        num_bits: usize,
    ) -> Self {
        let op = if let Some(native) =
            NativeOp::try_from_serial_optype(serial_op.op_type.clone(), num_qubits, num_bits)
        {
            Tk1Op::Native(native)
        } else {
            Tk1Op::Opaque(OpaqueTk1Op::new_from_op(serial_op, num_qubits, num_bits))
        };
        debug_assert_eq!(num_qubits, op.qubit_inputs().max(op.qubit_outputs()));
        debug_assert_eq!(num_bits, op.bit_inputs().max(op.bit_outputs()));
        op
    }

    /// Get the hugr optype for the operation.
    pub fn optype(&self) -> OpType {
        match self {
            Tk1Op::Native(native_op) => native_op.optype().clone(),
            Tk1Op::Opaque(json_op) => json_op.as_extension_op().into(),
        }
    }

    /// Consumes the operation and returns a hugr optype.
    pub fn into_optype(self) -> OpType {
        match self {
            Tk1Op::Native(native_op) => native_op.into_optype(),
            Tk1Op::Opaque(json_op) => json_op.as_extension_op().into(),
        }
    }

    /// Returns the ports corresponding to parameters for this operation.
    pub fn param_ports(&self) -> impl Iterator<Item = IncomingPort> + '_ {
        match self {
            Tk1Op::Native(native_op) => itertools::Either::Left(native_op.param_ports()),
            Tk1Op::Opaque(json_op) => itertools::Either::Right(json_op.param_ports()),
        }
    }

    /// Returns the number of qubit inputs for this operation.
    pub fn qubit_inputs(&self) -> usize {
        match self {
            Tk1Op::Native(native_op) => native_op.input_qubits,
            Tk1Op::Opaque(json_op) => json_op.num_qubits,
        }
    }

    /// Returns the number of bit inputs for this operation.
    pub fn bit_inputs(&self) -> usize {
        match self {
            Tk1Op::Native(native_op) => native_op.input_bits,
            Tk1Op::Opaque(json_op) => json_op.num_bits,
        }
    }

    /// Returns the number of qubit outputs for this operation.
    pub fn qubit_outputs(&self) -> usize {
        match self {
            Tk1Op::Native(native_op) => native_op.output_qubits,
            Tk1Op::Opaque(json_op) => json_op.num_qubits,
        }
    }

    /// Returns the number of bit outputs for this operation.
    pub fn bit_outputs(&self) -> usize {
        match self {
            Tk1Op::Native(native_op) => native_op.output_bits,
            Tk1Op::Opaque(json_op) => json_op.num_bits,
        }
    }

    /// Returns the number of parameters for this operation.
    pub fn num_params(&self) -> usize {
        match self {
            Tk1Op::Native(native_op) => native_op.num_params,
            Tk1Op::Opaque(json_op) => json_op.num_params,
        }
    }
}

impl From<Tk1Op> for OpType {
    fn from(tk1_op: Tk1Op) -> Self {
        tk1_op.into_optype()
    }
}

impl From<&Tk1Op> for OpType {
    fn from(tk1_op: &Tk1Op) -> Self {
        tk1_op.optype()
    }
}
