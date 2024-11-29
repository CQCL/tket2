//! Serialization and deserialization of circuits using the `pytket` JSON format.

mod decoder;
mod encoder;
mod op;
mod param;

use hugr::types::Type;

use hugr::Node;
use itertools::Itertools;
// Required for serialising ops in the tket1 hugr extension.
pub(crate) use op::serialised::OpaqueTk1Op;

#[cfg(test)]
mod tests;

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::path::Path;
use std::{fs, io};

use hugr::ops::{NamedOp, OpType};

use derive_more::{Display, Error, From};
use tket_json_rs::circuit_json::SerialCircuit;
use tket_json_rs::optype::OpType as SerialOpType;
use tket_json_rs::register::{Bit, ElementId, Qubit};

use crate::circuit::Circuit;

use self::decoder::Tk1Decoder;
use self::encoder::Tk1Encoder;

pub use crate::passes::pytket::lower_to_pytket;

/// Prefix used for storing metadata in the hugr nodes.
pub const METADATA_PREFIX: &str = "TKET1";
/// The global phase specified as metadata.
const METADATA_PHASE: &str = "TKET1.phase";
/// Explicit names for the input qubit registers.
const METADATA_Q_REGISTERS: &str = "TKET1.qubit_registers";
/// The reordered qubit registers in the output, if an implicit permutation was applied.
const METADATA_Q_OUTPUT_REGISTERS: &str = "TKET1.qubit_output_registers";
/// Explicit names for the input bit registers.
const METADATA_B_REGISTERS: &str = "TKET1.bit_registers";
/// The reordered bit registers in the output, if an implicit permutation was applied.
const METADATA_B_OUTPUT_REGISTERS: &str = "TKET1.bit_output_registers";
/// A tket1 operation "opgroup" field.
const METADATA_OPGROUP: &str = "TKET1.opgroup";
/// Explicit names for the input parameter wires.
const METADATA_INPUT_PARAMETERS: &str = "TKET1.input_parameters";

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
        let mut decoder = Tk1Decoder::try_new(&self)?;

        if !self.phase.is_empty() {
            // TODO - add a phase gate
            // let phase = Param::new(serialcirc.phase);
            // decoder.add_phase(phase);
        }

        for com in self.commands {
            decoder.add_command(com)?;
        }
        Ok(decoder.finish().into())
    }

    fn encode(circ: &Circuit) -> Result<Self, Self::EncodeError> {
        let mut encoder = Tk1Encoder::new(circ)?;
        for com in circ.commands() {
            let optype = com.optype();
            encoder.add_command(com.clone(), optype)?;
        }
        Ok(encoder.finish(circ))
    }
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
///
/// You may need to normalize the circuit using [`lower_to_pytket`] before saving.
///
/// # Errors
///
/// Returns an error if the circuit is not flat or if it contains operations not
/// supported by pytket.
pub fn save_tk1_json_file(circ: &Circuit, path: impl AsRef<Path>) -> Result<(), TK1ConvertError> {
    let file = fs::File::create(path)?;
    let writer = io::BufWriter::new(file);
    save_tk1_json_writer(circ, writer)
}

/// Save a circuit in TK1 JSON format to a writer.
///
/// You may need to normalize the circuit using [`lower_to_pytket`] before saving.
///
/// # Errors
///
/// Returns an error if the circuit is not flat or if it contains operations not
/// supported by pytket.
pub fn save_tk1_json_writer(circ: &Circuit, w: impl io::Write) -> Result<(), TK1ConvertError> {
    let serial_circ = SerialCircuit::encode(circ)?;
    serde_json::to_writer(w, &serial_circ)?;
    Ok(())
}

/// Save a circuit in TK1 JSON format to a String.
///
/// You may need to normalize the circuit using [`lower_to_pytket`] before saving.
///
/// # Errors
///
/// Returns an error if the circuit is not flat or if it contains operations not
/// supported by pytket.
pub fn save_tk1_json_str(circ: &Circuit) -> Result<String, TK1ConvertError> {
    let mut buf = io::BufWriter::new(Vec::new());
    save_tk1_json_writer(circ, &mut buf)?;
    let bytes = buf.into_inner().unwrap();
    Ok(String::from_utf8(bytes)?)
}

/// Error type for conversion between `Op` and `OpType`.
#[derive(Display, Debug, Error, From)]
#[non_exhaustive]
pub enum OpConvertError {
    /// The serialized operation is not supported.
    #[display("Unsupported serialized pytket operation: {_0:?}")]
    #[error(ignore)] // `_0` is not the error source
    UnsupportedSerializedOp(SerialOpType),
    /// The serialized operation is not supported.
    #[display("Cannot serialize tket2 operation: {_0:?}")]
    #[error(ignore)] // `_0` is not the error source
    UnsupportedOpSerialization(OpType),
    /// The operation has non-serializable inputs.
    #[display("Operation {} in {node} has an unsupported input of type {typ}.", optype.name())]
    UnsupportedInputType {
        /// The unsupported type.
        typ: Type,
        /// The operation name.
        optype: OpType,
        /// The node.
        node: Node,
    },
    /// The operation has non-serializable outputs.
    #[display("Operation {} in {node} has an unsupported output of type {typ}.", optype.name())]
    UnsupportedOutputType {
        /// The unsupported type.
        typ: Type,
        /// The operation name.
        optype: OpType,
        /// The node.
        node: Node,
    },
    /// A parameter input could not be evaluated.
    #[display("The {typ} parameter input for operation {} in {node} could not be resolved.", optype.name())]
    UnresolvedParamInput {
        /// The parameter type.
        typ: Type,
        /// The operation with the missing input param.
        optype: OpType,
        /// The node.
        node: Node,
    },
    /// The operation has output-only qubits.
    /// This is not currently supported by the encoder.
    #[display("Operation {} in {node} has more output qubits than inputs.", optype.name())]
    TooManyOutputQubits {
        /// The unsupported type.
        typ: Type,
        /// The operation name.
        optype: OpType,
        /// The node.
        node: Node,
    },
    /// The opaque tket1 operation had an invalid type parameter.
    #[display("Opaque TKET1 operation had an invalid type parameter. {error}")]
    #[from]
    InvalidOpaqueTypeParam {
        /// The serialization error.
        #[error(source)]
        error: serde_json::Error,
    },
    /// Tried to decode a tket1 operation with not enough parameters.
    #[display(
        "Operation {} is missing encoded parameters. Expected at least {expected} but only \"{}\" were specified.",
        optype.name(),
        params.iter().join(", "),
    )]
    MissingSerialisedParams {
        /// The operation name.
        optype: OpType,
        /// The expected number of parameters.
        expected: usize,
        /// The given of parameters.
        params: Vec<String>,
    },
    /// Tried to decode a tket1 operation with not enough qubit/bit arguments.
    #[display(
        "Operation {} is missing encoded arguments. Expected {expected_qubits} and {expected_bits}, but only \"{args:?}\" were specified.",
        optype.name(),
    )]
    MissingSerialisedArguments {
        /// The operation name.
        optype: OpType,
        /// The expected number of qubits.
        expected_qubits: usize,
        /// The expected number of bits.
        expected_bits: usize,
        /// The given of parameters.
        args: Vec<ElementId>,
    },
}

/// Error type for conversion between `Op` and `OpType`.
#[derive(Debug, Display, Error, From)]
#[non_exhaustive]
pub enum TK1ConvertError {
    /// Operation conversion error.
    #[from]
    OpConversionError(OpConvertError),
    /// The circuit has non-serializable inputs.
    #[display("Circuit contains non-serializable input of type {typ}.")]
    NonSerializableInputs {
        /// The unsupported type.
        typ: Type,
    },
    /// The circuit uses multi-indexed registers.
    //
    // This could be supported in the future, if there is a need for it.
    #[display("Register {register} in the circuit has multiple indices. Tket2 does not support multi-indexed registers.")]
    MultiIndexedRegister {
        /// The register name.
        register: String,
    },
    /// Invalid JSON,
    #[display("Invalid pytket JSON. {_0}")]
    #[from]
    InvalidJson(serde_json::Error),
    /// Invalid JSON,
    #[display("Invalid JSON encoding. {_0}")]
    #[from]
    InvalidJsonEncoding(std::string::FromUtf8Error),
    /// File not found.,
    #[display("Unable to load pytket json file. {_0}")]
    #[from]
    FileLoadError(io::Error),
}

/// A hashed register, used to identify registers in the [`Tk1Decoder::register_wire`] map,
/// avoiding string and vector clones on lookup.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
struct RegisterHash {
    hash: u64,
}

impl From<&ElementId> for RegisterHash {
    fn from(reg: &ElementId) -> Self {
        let mut hasher = DefaultHasher::new();
        reg.hash(&mut hasher);
        Self {
            hash: hasher.finish(),
        }
    }
}

impl From<&Qubit> for RegisterHash {
    fn from(reg: &Qubit) -> Self {
        let mut hasher = DefaultHasher::new();
        reg.hash(&mut hasher);
        Self {
            hash: hasher.finish(),
        }
    }
}

impl From<&Bit> for RegisterHash {
    fn from(reg: &Bit) -> Self {
        let mut hasher = DefaultHasher::new();
        reg.hash(&mut hasher);
        Self {
            hash: hasher.finish(),
        }
    }
}
