//! Json serialization and deserialization.

mod decoder;
mod encoder;
pub mod op;

#[cfg(test)]
mod tests;

use std::path::Path;
use std::{fs, io};

use hugr::ops::OpType;
use hugr::std_extensions::arithmetic::float_types::ConstF64;
use hugr::values::Value;
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

/// A JSON-serialized circuit that can be converted to a [`Hugr`].
pub trait TKETDecode: Sized {
    /// The error type for decoding.
    type DecodeError;
    /// The error type for decoding.
    type EncodeError;
    /// Convert the serialized circuit to a [`Hugr`].
    fn decode(self) -> Result<Hugr, Self::DecodeError>;
    /// Convert a [`Hugr`] to a new serialized circuit.
    fn encode<'circ>(circuit: &'circ impl Circuit<'circ>) -> Result<Self, Self::EncodeError>;
}

impl TKETDecode for SerialCircuit {
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
        for com in circ.commands() {
            let optype = circ.command_optype(&com);
            encoder.add_command(com, optype)?;
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

/// Load a TKET1 circuit from a JSON file.
pub fn load_tk1_json_file(path: impl AsRef<Path>) -> Result<Hugr, TK1LoadError> {
    let file = fs::File::open(path)?;
    let reader = io::BufReader::new(file);
    let ser: SerialCircuit = serde_json::from_reader(reader)?;
    Ok(ser.decode()?)
}

/// Load a TKET1 circuit from a JSON string.
pub fn load_tk1_json_str(json: &str) -> Result<Hugr, TK1LoadError> {
    let ser: SerialCircuit = serde_json::from_str(json)?;
    Ok(ser.decode()?)
}

/// Error type for conversion between `Op` and `OpType`.
#[derive(Debug, Error)]
pub enum TK1LoadError {
    /// The serialized operation is not supported.
    #[error("unsupported serialized operation: {0:?}")]
    UnsupportedSerializedOp(JsonOpType),
    /// The serialized operation is not supported.
    #[error("cannot serialize operation: {0:?}")]
    UnsupportedOpSerialization(OpType),
    /// The serialized operation is not supported.
    #[error("cannot serialize operation: {0:?}")]
    NonSerializableInputs(OpType),
    /// Invalid JSON,
    #[error("invalid JSON")]
    InvalidJson,
    /// File not found.,
    #[error("unable to load file")]
    FileLoadError,
}

impl From<serde_json::Error> for TK1LoadError {
    fn from(_: serde_json::Error) -> Self {
        Self::InvalidJson
    }
}

impl From<io::Error> for TK1LoadError {
    fn from(_: io::Error) -> Self {
        Self::FileLoadError
    }
}

impl From<OpConvertError> for TK1LoadError {
    fn from(value: OpConvertError) -> Self {
        match value {
            OpConvertError::UnsupportedSerializedOp(op) => Self::UnsupportedSerializedOp(op),
            OpConvertError::UnsupportedOpSerialization(op) => Self::UnsupportedOpSerialization(op),
            OpConvertError::NonSerializableInputs(op) => Self::NonSerializableInputs(op),
        }
    }
}

/// Try to interpret a TKET1 parameter as a constant value.
#[inline]
fn try_param_to_constant(param: &str) -> Option<Value> {
    if let Ok(f) = param.parse::<f64>() {
        Some(ConstF64::new(f).into())
    } else if param.split('/').count() == 2 {
        // TODO: Use the rational types from `Hugr::extensions::rotation`
        let (n, d) = param.split_once('/').unwrap();
        let n = n.parse::<f64>().unwrap();
        let d = d.parse::<f64>().unwrap();
        Some(ConstF64::new(n / d).into())
    } else {
        None
    }
}
