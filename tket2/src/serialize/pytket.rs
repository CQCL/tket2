//! Serialization and deserialization of circuits using the `pytket` JSON format.

mod decoder;
mod encoder;
mod op;

use hugr::types::Type;
pub(crate) use op::JsonOp;

#[cfg(test)]
mod tests;

use hugr::CircuitUnit;

use std::path::Path;
use std::{fs, io};

use hugr::ops::{OpType, Value};
use hugr::std_extensions::arithmetic::float_types::{ConstF64, FLOAT64_TYPE};

use thiserror::Error;
use tket_json_rs::circuit_json::SerialCircuit;
use tket_json_rs::optype::OpType as JsonOpType;

use crate::circuit::Circuit;

use self::decoder::JsonDecoder;
use self::encoder::JsonEncoder;

/// Prefix used for storing metadata in the hugr nodes.
pub const METADATA_PREFIX: &str = "TKET1_JSON";
/// The global phase specified as metadata.
const METADATA_PHASE: &str = "TKET1_JSON.phase";
/// The implicit permutation of qubits.
const METADATA_IMPLICIT_PERM: &str = "TKET1_JSON.implicit_permutation";
/// Explicit names for the input qubit registers.
const METADATA_Q_REGISTERS: &str = "TKET1_JSON.qubit_registers";
/// Explicit names for the input bit registers.
const METADATA_B_REGISTERS: &str = "TKET1_JSON.bit_registers";

/// A serialized representation of a [`Circuit`].
///
/// Implemented by [`SerialCircuit`], the JSON format used by tket1's `pytket` library.
pub trait TKETDecode: Sized {
    /// The error type for decoding.
    type DecodeError;
    /// The error type for decoding.
    type EncodeError;
    /// Convert the serialized circuit to a circuit.
    fn decode(self) -> Result<Circuit, Self::DecodeError>;
    /// Convert a circuit to a new serialized circuit.
    fn encode(circuit: &Circuit) -> Result<Self, Self::EncodeError>;
}

impl TKETDecode for SerialCircuit {
    type DecodeError = TK1ConvertError;
    type EncodeError = TK1ConvertError;

    fn decode(self) -> Result<Circuit, Self::DecodeError> {
        let mut decoder = JsonDecoder::new(&self);

        if !self.phase.is_empty() {
            // TODO - add a phase gate
            // let phase = Param::new(serialcirc.phase);
            // decoder.add_phase(phase);
        }

        for com in self.commands {
            decoder.add_command(com);
        }
        Ok(decoder.finish().into())
    }

    fn encode(circ: &Circuit) -> Result<Self, Self::EncodeError> {
        let mut encoder = JsonEncoder::new(circ);
        let f64_inputs = circ.units().filter_map(|(wire, _, t)| match (wire, t) {
            (CircuitUnit::Wire(wire), t) if t == FLOAT64_TYPE => Some(Ok(wire)),
            (CircuitUnit::Linear(_), _) => None,
            (_, typ) => Some(Err(TK1ConvertError::NonSerializableInputs { typ })),
        });
        for (i, wire) in f64_inputs.enumerate() {
            let wire = wire?;
            let param = format!("f{i}");
            encoder.add_parameter(wire, param);
        }
        for com in circ.commands() {
            let optype = com.optype();
            encoder.add_command(com.clone(), optype)?;
        }
        Ok(encoder.finish())
    }
}

/// Error type for conversion between `Op` and `OpType`.
#[derive(Debug, Error)]
pub enum OpConvertError {
    /// The serialized operation is not supported.
    #[error("Unsupported serialized pytket operation: {0:?}")]
    UnsupportedSerializedOp(JsonOpType),
    /// The serialized operation is not supported.
    #[error("Cannot serialize tket2 operation: {0:?}")]
    UnsupportedOpSerialization(OpType),
}

/// Load a TKET1 circuit from a JSON file.
pub fn load_tk1_json_file(path: impl AsRef<Path>) -> Result<Circuit, TK1ConvertError> {
    let file = fs::File::open(path)?;
    let reader = io::BufReader::new(file);
    load_tk1_json_reader(reader)
}

/// Load a TKET1 circuit from a JSON reader.
pub fn load_tk1_json_reader(json: impl io::Read) -> Result<Circuit, TK1ConvertError> {
    let ser: SerialCircuit = serde_json::from_reader(json)?;
    let circ: Circuit = ser.decode()?;
    Ok(circ)
}

/// Load a TKET1 circuit from a JSON string.
pub fn load_tk1_json_str(json: &str) -> Result<Circuit, TK1ConvertError> {
    let reader = json.as_bytes();
    load_tk1_json_reader(reader)
}

/// Save a circuit to file in TK1 JSON format.
pub fn save_tk1_json_file(circ: &Circuit, path: impl AsRef<Path>) -> Result<(), TK1ConvertError> {
    let file = fs::File::create(path)?;
    let writer = io::BufWriter::new(file);
    save_tk1_json_writer(circ, writer)
}

/// Save a circuit in TK1 JSON format to a writer.
pub fn save_tk1_json_writer(circ: &Circuit, w: impl io::Write) -> Result<(), TK1ConvertError> {
    let serial_circ = SerialCircuit::encode(circ)?;
    serde_json::to_writer(w, &serial_circ)?;
    Ok(())
}

/// Save a circuit in TK1 JSON format to a String.
pub fn save_tk1_json_str(circ: &Circuit) -> Result<String, TK1ConvertError> {
    let mut buf = io::BufWriter::new(Vec::new());
    save_tk1_json_writer(circ, &mut buf)?;
    let bytes = buf.into_inner().unwrap();
    Ok(String::from_utf8(bytes)?)
}

/// Error type for conversion between `Op` and `OpType`.
#[derive(Debug, Error)]
pub enum TK1ConvertError {
    /// Operation conversion error.
    #[error("{0}")]
    OpConversionError(#[from] OpConvertError),
    /// The circuit has non-serializable inputs.
    #[error("Circuit contains non-serializable input of type {typ}.")]
    NonSerializableInputs {
        /// The unsupported type.
        typ: Type,
    },
    /// Invalid JSON,
    #[error("Invalid pytket JSON. {0}")]
    InvalidJson(#[from] serde_json::Error),
    /// Invalid JSON,
    #[error("Invalid JSON encoding. {0}")]
    InvalidJsonEncoding(#[from] std::string::FromUtf8Error),
    /// File not found.,
    #[error("Unable to load pytket json file. {0}")]
    FileLoadError(#[from] io::Error),
}

#[inline]
fn parse_val(n: &str) -> Option<f64> {
    n.parse::<f64>().ok()
}
/// Try to interpret a TKET1 parameter as a constant value.
#[inline]
fn try_param_to_constant(param: &str) -> Option<Value> {
    if let Some(f) = parse_val(param) {
        Some(ConstF64::new(f).into())
    } else if param.split('/').count() == 2 {
        // TODO: Use the rational types from `Hugr::extensions::rotation`
        let (n, d) = param.split_once('/').unwrap();
        let n = parse_val(n)?;
        let d = parse_val(d)?;
        Some(ConstF64::new(n / d).into())
    } else {
        None
    }
}