//! Serialization and deserialization of circuits using the `pytket` JSON format.

mod decoder;
pub mod encoder;
pub mod extension;

pub use encoder::{default_encoder_config, Tk1EncoderConfig, Tk1EncoderContext};
pub use extension::PytketEmitter;

use hugr::core::HugrNode;

use hugr::{Hugr, Wire};
use itertools::Itertools;

#[cfg(test)]
mod tests;

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::path::Path;
use std::{fs, io};

use hugr::ops::OpType;

use derive_more::{Display, Error, From};
use tket_json_rs::circuit_json::SerialCircuit;
use tket_json_rs::register::{Bit, ElementId, Qubit};

use crate::circuit::Circuit;

use self::decoder::Tk1DecoderContext;

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
    /// Convert a circuit to a serialized pytket circuit.
    ///
    /// Uses a default set of emitters to translate operations.
    /// If the circuit contains non-std operations or types,
    /// use [`TKETDecode::encode_with_config`] instead.
    fn encode(circuit: &Circuit) -> Result<Self, Self::EncodeError>;
    /// Convert a circuit to a serialized pytket circuit.
    ///
    /// You may use [`TKETDecode::encode`] if the circuit does not contain
    /// non-std operations or types.
    fn encode_with_config(
        circuit: &Circuit,
        config: Tk1EncoderConfig<Hugr>,
    ) -> Result<Self, Self::EncodeError>;
}

impl TKETDecode for SerialCircuit {
    type DecodeError = Tk1ConvertError;
    type EncodeError = Tk1ConvertError;

    fn decode(self) -> Result<Circuit, Self::DecodeError> {
        let mut decoder = Tk1DecoderContext::try_new(&self)?;

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

    fn encode(circuit: &Circuit) -> Result<Self, Self::EncodeError> {
        let config = default_encoder_config();
        Self::encode_with_config(circuit, config)
    }

    fn encode_with_config(
        circuit: &Circuit,
        config: Tk1EncoderConfig<Hugr>,
    ) -> Result<Self, Self::EncodeError> {
        let mut encoder = Tk1EncoderContext::new(circuit, circuit.parent(), config)?;
        encoder.run_encoder(circuit, circuit.parent())?;
        let (serial, _) = encoder.finish(circuit, circuit.parent())?;
        Ok(serial)
    }
}

/// Load a TKET1 circuit from a JSON file.
pub fn load_tk1_json_file(path: impl AsRef<Path>) -> Result<Circuit, Tk1ConvertError> {
    let file = fs::File::open(path)?;
    let reader = io::BufReader::new(file);
    load_tk1_json_reader(reader)
}

/// Load a TKET1 circuit from a JSON reader.
pub fn load_tk1_json_reader(json: impl io::Read) -> Result<Circuit, Tk1ConvertError> {
    let ser: SerialCircuit = serde_json::from_reader(json)?;
    let circ: Circuit = ser.decode()?;
    Ok(circ)
}

/// Load a TKET1 circuit from a JSON string.
pub fn load_tk1_json_str(json: &str) -> Result<Circuit, Tk1ConvertError> {
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
pub fn save_tk1_json_file(circ: &Circuit, path: impl AsRef<Path>) -> Result<(), Tk1ConvertError> {
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
pub fn save_tk1_json_writer(circ: &Circuit, w: impl io::Write) -> Result<(), Tk1ConvertError> {
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
pub fn save_tk1_json_str(circ: &Circuit) -> Result<String, Tk1ConvertError> {
    let mut buf = io::BufWriter::new(Vec::new());
    save_tk1_json_writer(circ, &mut buf)?;
    let bytes = buf.into_inner().unwrap();
    Ok(String::from_utf8(bytes)?)
}

/// Error type for conversion between pytket operations and tket2 ops.
#[derive(Display, derive_more::Debug, Error)]
#[non_exhaustive]
#[debug(bounds(N: HugrNode))]
pub enum OpConvertError<N = hugr::Node> {
    /// The serialized operation is not supported.
    #[display("Cannot serialize tket2 operation: {op}")]
    UnsupportedOpSerialization {
        /// The operation.
        op: OpType,
    },
    /// Tried to decode a tket1 operation with not enough parameters.
    #[display(
        "Operation {} is missing encoded parameters. Expected at least {expected} but only \"{}\" were specified.",
        optype,
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
        optype,
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
    /// Tried to query the values associated with an unexplored wire.
    ///
    /// This reflects a bug in the operation encoding logic of an operation.
    #[display("Could not find values associated with wire {wire}.")]
    WireHasNoValues {
        /// The wire that has no values.
        wire: Wire<N>,
    },
    /// Tried to add values to an already registered wire.
    ///
    /// This reflects a bug in the operation encoding logic of an operation.
    #[display("Tried to register values for wire {wire}, but it already has associated values.")]
    WireAlreadyHasValues {
        /// The wire that already has values.
        wire: Wire<N>,
    },
}

/// Error type for conversion between tket2 ops and pytket operations.
#[derive(derive_more::Debug, Display, Error, From)]
#[non_exhaustive]
#[debug(bounds(N: HugrNode))]
pub enum Tk1ConvertError<N = hugr::Node> {
    /// Tried to encode a non-dataflow region.
    #[display("Cannot encode non-dataflow region at {region} with type {optype}.")]
    NonDataflowRegion {
        /// The region that is not a dataflow region.
        region: N,
        /// The operation type of the region.
        optype: String,
    },
    /// Operation conversion error.
    #[from]
    OpConversionError(OpConvertError<N>),
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
    /// Custom user-defined error raised while encoding an operation.
    #[display("Error while encoding operation: {msg}")]
    CustomError {
        /// The custom error message
        msg: String,
    },
}

impl<N> Tk1ConvertError<N> {
    /// Create a new error with a custom message.
    pub fn custom(msg: impl Into<String>) -> Self {
        Self::CustomError { msg: msg.into() }
    }
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
