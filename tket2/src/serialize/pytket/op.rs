//! This module defines the internal [`Tk1Op`] struct wrapping the logic for
//! going between `tket_json_rs::optype::OpType` and `hugr::ops::OpType`.
//!
//! The `Tk1Op` tries to homogenize the
//! `tket_json_rs::circuit_json::Operation`s coming from the encoded TKET1
//! circuits by ensuring they always define a signature, and computing the
//! explicit count of qubits and linear bits.

use std::borrow::Cow;

use hugr::extension::ExtensionId;
use hugr::ops::OpType;
use hugr::HugrView;

use super::encoder::Tk1EncoderContext;
use super::{OpConvertError, Tk1DecoderContext};

pub(crate) mod serialised;
mod tk2op;

/// An encoder of HUGR operations and types that transform them
/// into pytket primitives.
pub trait Tk1Decoder {
    /// Given a pytket operation, try to convert it to a HUGR operation of this type.
    fn op_from_pytket(
        &self,
        op: &tket_json_rs::circuit_json::Operation,
        encoder: &Tk1DecoderContext,
    ) -> Result<bool, OpConvertError>;
}
