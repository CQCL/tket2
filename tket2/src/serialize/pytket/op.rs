//! This module defines the internal [`Tk1Op`] struct wrapping the logic for
//! going between `tket_json_rs::optype::OpType` and `hugr::ops::OpType`.
//!
//! The `Tk1Op` tries to homogenize the
//! `tket_json_rs::circuit_json::Operation`s coming from the encoded TKET1
//! circuits by ensuring they always define a signature, and computing the
//! explicit count of qubits and linear bits.

use hugr::extension::ExtensionId;
use hugr::ops::OpType;
use hugr::Node;

use super::encoder::Tk1EncoderContext;
use super::{OpConvertError, Tk1Decoder};

pub(crate) mod serialised;
mod tk2op;

/// An encoder/decoders of HUGR operations and types that transform them
/// into pytket primitives.
pub trait Tk1Encoder {
    /// The name of the extension this encoder/decoder is for.
    fn extension(&self) -> &ExtensionId;

    /// Given a node in the HUGR circuit and its operation type, try to convert
    /// it to a pytket operation and add it to the pytket encoder.
    ///
    /// Returns `true` if the operation was successfully converted. If that is
    /// the case, no further encoders will be called.
    ///
    /// If the operation is not supported by the encoder, return `false`.
    fn op_to_pytket(
        &self,
        node: Node,
        op: &OpType,
        encoder: &mut Tk1EncoderContext,
    ) -> Result<bool, OpConvertError> {
        Ok(false)
    }

    /// Given a HUGR type, return the number of qubits, bits, and sympy
    /// parameters of its pytket counterpart.
    ///
    /// If the type is not supported by the encoder, return `None`.
    fn type_to_pytket(&self, op: &OpType) -> Result<Option<RegisterCount>, OpConvertError> {
        Ok(None)
    }

    /// Given a pytket operation, try to convert it to a HUGR operation of this type.
    fn op_from_pytket(
        &self,
        op: &tket_json_rs::circuit_json::Operation,
        encoder: &Tk1Decoder,
    ) -> Result<bool, OpConvertError> {
        Ok(false)
    }
}

/// The number of pytket qubits, bits, and sympy parameters corresponding to a
/// HUGR type.
///
/// Used as return value for [`Tk1Encoder::type_to_pytket`].
#[derive(Clone, Copy, Debug, Default)]
pub struct RegisterCount {
    pub qubits: usize,
    pub bits: usize,
    pub params: usize,
}
