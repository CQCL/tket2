//! Json serialization and deserialization.

mod decoder;
mod encoder;
pub mod op;

#[cfg(test)]
mod tests;

use hugr::ops::{ConstValue, OpType};
use hugr::Hugr;

use thiserror::Error;
use tket_json_rs::circuit_json::SerialCircuit;
use tket_json_rs::optype::OpType as JsonOpType;

use crate::circuit::Circuit;

use self::decoder::JsonDecoder;
use self::encoder::JsonEncoder;

/// Prefix used for storing metadata in the hugr nodes.
pub const METADATA_PREFIX: &str = "TKET1_JSON";
const METADATA_PHASE: &str = "TKET1_JSON.phase";
const METADATA_IMPLICIT_PERM: &str = "TKET1_JSON.implicit_permutation";

/// A JSON-serialized TKET1 circuit that can be converted to a [`Hugr`].
pub trait TKET1Decode: Sized {
    /// The error type for decoding.
    type DecodeError;
    /// The error type for decoding.
    type EncodeError;
    /// Convert the serialized circuit to a [`Hugr`].
    fn decode(self) -> Result<Hugr, Self::DecodeError>;
    /// Convert a [`Hugr`] to a new serialized circuit.
    fn encode<'circ>(circuit: &'circ impl Circuit<'circ>) -> Result<Self, Self::EncodeError>;
}

impl TKET1Decode for SerialCircuit {
    type DecodeError = OpConvertError;
    type EncodeError = OpConvertError;

    fn decode(self) -> Result<Hugr, Self::DecodeError> {
        let mut decoder = JsonDecoder::new(&self);

        if !self.phase.is_empty() {
            // TODO - add a phase gate
            // let phase = Param::new(serialcirc.phase);
            // decoder.add_phase(phase);
        }

        for com in self.commands {
            decoder.add_command(com);
        }
        Ok(decoder.finish())
    }

    fn encode<'circ>(circ: &'circ impl Circuit<'circ>) -> Result<Self, Self::EncodeError> {
        let mut encoder = JsonEncoder::new(circ);

        // TODO Restore the global phase

        // TODO Restore the implicit permutation

        for com in circ.commands() {
            encoder.add_command(com)?;
        }
        Ok(encoder.finish())
    }
}

/// Error type for conversion between `Op` and `OpType`.
#[derive(Debug, Error)]
pub enum OpConvertError {
    /// The serialized operation is not supported.
    #[error("Unsupported serialized operation: {0:?}")]
    UnsupportedSerializedOp(JsonOpType),
    /// The serialized operation is not supported.
    #[error("Cannot serialize operation: {0:?}")]
    UnsupportedOpSerialization(OpType),
    /// The serialized operation is not supported.
    #[error("Cannot serialize operation: {0:?}")]
    NonSerializableInputs(OpType),
}

/// Try to interpret a TKET1 parameter as a constant value.
#[inline]
fn try_param_to_constant(param: &str) -> Option<ConstValue> {
    if let Ok(f) = param.parse::<f64>() {
        Some(ConstValue::F64(f))
    } else if param.split('/').count() == 2 {
        // TODO: Use the rational types from `Hugr::extensions::rotation`
        let (n, d) = param.split_once('/').unwrap();
        let n = n.parse::<f64>().unwrap();
        let d = d.parse::<f64>().unwrap();
        Some(ConstValue::F64(n / d))
    } else {
        None
    }
}
