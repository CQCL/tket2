//! Json serialization and deserialization.

mod decoder;
//mod encoder;
pub mod op;

#[cfg(test)]
mod tests;

use hugr::ops::OpType;
use hugr::Hugr;

use thiserror::Error;
use tket_json_rs::circuit_json::SerialCircuit;
use tket_json_rs::optype::OpType as JsonOpType;

use self::decoder::JsonDecoder;

/// A JSON-serialized TKET1 circuit that can be converted to a [`Hugr`].
pub trait TKET1Decode: Sized {
    /// The error type for decoding.
    type DecodeError;
    /// Convert the serialized circuit to a [`Hugr`].
    fn decode(self) -> Result<Hugr, Self::DecodeError>;
}

impl TKET1Decode for SerialCircuit {
    type DecodeError = OpConvertError;

    fn decode(self) -> Result<Hugr, Self::DecodeError> {
        let mut decoder = JsonDecoder::new(&self);

        if !self.phase.is_empty() {
            // TODO - add a phase gate
            // let phase = Param::new(serialcirc.phase);
            // decoder.add_phase(phase);
        }

        // TODO: Check the implicit permutation in the serialized circuit.

        for com in self.commands {
            decoder.add_command(com);
        }
        Ok(decoder.finish())
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
}
