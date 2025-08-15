//! Serialization and deserialization of circuits using the `pytket` JSON format.

mod config;
pub mod decoder;
pub mod encoder;
pub mod extension;

pub use config::{
    default_decoder_config, default_encoder_config, PytketDecoderConfig, PytketEncoderConfig,
    TypeTranslatorSet,
};
pub use encoder::PytketEncoderContext;
pub use extension::PytketEmitter;

use hugr::core::HugrNode;

use hugr::hugr::hugrmut::HugrMut;
use hugr::ops::handle::NodeHandle;
use hugr::{Hugr, Wire};
use itertools::Itertools;

#[cfg(test)]
mod tests;

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::path::Path;
use std::sync::Arc;
use std::{fs, io};

use hugr::ops::OpType;

use derive_more::{Display, Error, From};
use tket_json_rs::circuit_json::SerialCircuit;
use tket_json_rs::register::{Bit, ElementId, Qubit};

use crate::circuit::Circuit;
use crate::serialize::pytket::extension::RegisterCount;

use self::decoder::PytketDecoderContext;

pub use crate::passes::pytket::lower_to_pytket;

/// Prefix used for storing metadata in the hugr nodes.
pub const METADATA_PREFIX: &str = "TKET1";
/// The global phase specified as metadata.
pub const METADATA_PHASE: &str = "TKET1.phase";
/// Explicit names for the input qubit registers.
pub const METADATA_Q_REGISTERS: &str = "TKET1.qubit_registers";
/// The reordered qubit registers in the output, if an implicit permutation was applied.
pub const METADATA_Q_OUTPUT_REGISTERS: &str = "TKET1.qubit_output_registers";
/// Explicit names for the input bit registers.
pub const METADATA_B_REGISTERS: &str = "TKET1.bit_registers";
/// The reordered bit registers in the output, if an implicit permutation was applied.
pub const METADATA_B_OUTPUT_REGISTERS: &str = "TKET1.bit_output_registers";
/// A tket1 operation "opgroup" field.
pub const METADATA_OPGROUP: &str = "TKET1.opgroup";
/// Explicit names for the input parameter wires.
pub const METADATA_INPUT_PARAMETERS: &str = "TKET1.input_parameters";

/// A serialized representation of a [`Circuit`].
///
/// Implemented by [`SerialCircuit`], the JSON format used by tket1's `pytket` library.
pub trait TKETDecode: Sized {
    /// Error type of decoding errors.
    type DecodeError;
    /// Error type of encoding errors.
    type EncodeError;
    /// Convert the serialized circuit to a circuit.
    ///
    /// Uses a default set of extension decoders to translate operations.
    fn decode(self) -> Result<Circuit, Self::DecodeError>;
    /// Convert the serialized circuit to a circuit.
    fn decode_with_config(
        self,
        config: impl Into<Arc<PytketDecoderConfig>>,
    ) -> Result<Circuit, Self::DecodeError>;
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
        config: impl Into<Arc<PytketEncoderConfig<Hugr>>>,
    ) -> Result<Self, Self::EncodeError>;
}

impl TKETDecode for SerialCircuit {
    type DecodeError = PytketDecodeError;
    type EncodeError = PytketEncodeError;

    fn decode(self) -> Result<Circuit, Self::DecodeError> {
        let config = default_decoder_config();
        Self::decode_with_config(self, config)
    }

    fn decode_with_config(
        self,
        config: impl Into<Arc<PytketDecoderConfig>>,
    ) -> Result<Circuit, Self::DecodeError> {
        let mut hugr = Hugr::new();

        let mut decoder =
            PytketDecoderContext::new(&self, &mut hugr, None, None, Vec::new(), config)?;
        decoder.run_decoder(self.commands)?;
        let main_func = decoder.finish()?;
        hugr.set_entrypoint(main_func.node());
        Ok(hugr.into())
    }

    fn encode(circuit: &Circuit) -> Result<Self, Self::EncodeError> {
        let config = default_encoder_config();
        Self::encode_with_config(circuit, config)
    }

    fn encode_with_config(
        circuit: &Circuit,
        config: impl Into<Arc<PytketEncoderConfig<Hugr>>>,
    ) -> Result<Self, Self::EncodeError> {
        let mut encoder = PytketEncoderContext::new(circuit, circuit.parent(), config)?;
        encoder.run_encoder(circuit, circuit.parent())?;
        let (serial, _) = encoder.finish(circuit, circuit.parent())?;
        Ok(serial)
    }
}

/// Load a TKET1 circuit from a JSON file.
///
/// If a decoder config is provided, it will be used to decode the circuit.
/// Otherwise, it defaults to [`default_decoder_config`].
pub fn load_tk1_json_file(
    path: impl AsRef<Path>,
    config: Option<PytketDecoderConfig>,
) -> Result<Circuit, PytketDecodeError> {
    let file = fs::File::open(path).map_err(PytketDecodeError::custom)?;
    let reader = io::BufReader::new(file);
    load_tk1_json_reader(reader, config)
}

/// Load a TKET1 circuit from a JSON reader.
///
/// If a decoder config is provided, it will be used to decode the circuit.
/// Otherwise, it defaults to [`default_decoder_config`].
pub fn load_tk1_json_reader(
    json: impl io::Read,
    config: Option<PytketDecoderConfig>,
) -> Result<Circuit, PytketDecodeError> {
    let config = config.unwrap_or_else(default_decoder_config);
    let ser: SerialCircuit = serde_json::from_reader(json).map_err(PytketDecodeError::custom)?;
    let circ: Circuit = ser.decode_with_config(config)?;
    Ok(circ)
}

/// Load a TKET1 circuit from a JSON string.
///
/// If a decoder config is provided, it will be used to decode the circuit.
/// Otherwise, it defaults to [`default_decoder_config`].
pub fn load_tk1_json_str(
    json: &str,
    config: Option<PytketDecoderConfig>,
) -> Result<Circuit, PytketDecodeError> {
    let reader = json.as_bytes();
    load_tk1_json_reader(reader, config)
}

/// Save a circuit to file in TK1 JSON format.
///
/// You may need to normalize the circuit using [`lower_to_pytket`] before saving.
///
/// If an encoder config is provided, it will be used to encode the circuit.
/// Otherwise, it defaults to [`default_encoder_config`].
///
/// # Errors
///
/// Returns an error if the circuit is not flat or if it contains operations not
/// supported by pytket.
pub fn save_tk1_json_file(
    circ: &Circuit,
    path: impl AsRef<Path>,
    config: Option<PytketEncoderConfig<Hugr>>,
) -> Result<(), PytketEncodeError> {
    let file = fs::File::create(path).map_err(PytketEncodeError::custom)?;
    let writer = io::BufWriter::new(file);
    save_tk1_json_writer(circ, writer, config)
}

/// Save a circuit in TK1 JSON format to a writer.
///
/// You may need to normalize the circuit using [`lower_to_pytket`] before saving.
///
/// If an encoder config is provided, it will be used to encode the circuit.
/// Otherwise, it defaults to [`default_encoder_config`].
///
/// # Errors
///
/// Returns an error if the circuit is not flat or if it contains operations not
/// supported by pytket.
pub fn save_tk1_json_writer(
    circ: &Circuit,
    w: impl io::Write,
    config: Option<PytketEncoderConfig<Hugr>>,
) -> Result<(), PytketEncodeError> {
    let config = config.unwrap_or_else(default_encoder_config);
    let serial_circ = SerialCircuit::encode_with_config(circ, config)?;
    serde_json::to_writer(w, &serial_circ).map_err(PytketEncodeError::custom)?;
    Ok(())
}

/// Save a circuit in TK1 JSON format to a String.
///
/// You may need to normalize the circuit using [`lower_to_pytket`] before saving.
///
/// If an encoder config is provided, it will be used to encode the circuit.
/// Otherwise, it defaults to [`default_encoder_config`].
///
/// # Errors
///
/// Returns an error if the circuit is not flat or if it contains operations not
/// supported by pytket.
pub fn save_tk1_json_str(
    circ: &Circuit,
    config: Option<PytketEncoderConfig<Hugr>>,
) -> Result<String, PytketEncodeError> {
    let mut buf = io::BufWriter::new(Vec::new());
    save_tk1_json_writer(circ, &mut buf, config)?;
    let bytes = buf.into_inner().unwrap();
    String::from_utf8(bytes).map_err(PytketEncodeError::custom)
}

/// Error type for conversion between pytket operations and tket ops.
#[derive(Display, derive_more::Debug, Error)]
#[non_exhaustive]
#[debug(bounds(N: HugrNode))]
pub enum OpConvertError<N = hugr::Node> {
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

/// Error type for conversion between tket ops and pytket operations.
#[derive(derive_more::Debug, Display, Error, From)]
#[non_exhaustive]
#[debug(bounds(N: HugrNode))]
pub enum PytketEncodeError<N = hugr::Node> {
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
    /// Custom user-defined error raised while encoding an operation.
    #[display("Error while encoding operation: {msg}")]
    CustomError {
        /// The custom error message
        msg: String,
    },
}

impl<N> PytketEncodeError<N> {
    /// Create a new error with a custom message.
    pub fn custom(msg: impl ToString) -> Self {
        Self::CustomError {
            msg: msg.to_string(),
        }
    }
}

/// Error type for conversion between tket2 ops and pytket operations.
#[derive(derive_more::Debug, Display, Error, Clone)]
#[non_exhaustive]
#[display(
    "{inner}{context}",
    context = {
        match (pytket_op, hugr_op) {
            (Some(pytket_op), Some(hugr_op)) => format!(". While decoding a pytket {pytket_op} as a hugr {hugr_op}"),
            (Some(pytket_op), None) => format!(". While decoding a pytket {pytket_op}"),
            (None, Some(hugr_op)) => format!(". While decoding a hugr {hugr_op}"),
            (None, None) => String::new(),
        }
    },
)]
pub struct PytketDecodeError {
    /// The kind of error.
    pub inner: PytketDecodeErrorInner,
    /// The pytket operation that caused the error, if applicable.
    pub pytket_op: Option<String>,
    /// The hugr operation that caused the error, if applicable.
    pub hugr_op: Option<String>,
}

impl PytketDecodeError {
    /// Create a new error with a custom message.
    pub fn custom(msg: impl ToString) -> Self {
        PytketDecodeErrorInner::CustomError {
            msg: msg.to_string(),
        }
        .into()
    }

    /// Create an error for an unknown qubit register.
    pub fn unknown_qubit_reg(register: &tket_json_rs::register::ElementId) -> Self {
        PytketDecodeErrorInner::UnknownQubitRegister {
            register: register.to_string(),
        }
        .into()
    }

    /// Create an error for an unknown bit register.
    pub fn unknown_bit_reg(register: &tket_json_rs::register::ElementId) -> Self {
        PytketDecodeErrorInner::UnknownBitRegister {
            register: register.to_string(),
        }
        .into()
    }

    /// Add the pytket operation name to the error.
    pub fn pytket_op(mut self, op: &tket_json_rs::OpType) -> Self {
        self.pytket_op = Some(format!("{op:?}"));
        self
    }

    /// Add the hugr operation name to the error.
    pub fn hugr_op(mut self, op: impl ToString) -> Self {
        self.hugr_op = Some(op.to_string());
        self
    }
}

impl From<PytketDecodeErrorInner> for PytketDecodeError {
    fn from(inner: PytketDecodeErrorInner) -> Self {
        Self {
            inner,
            pytket_op: None,
            hugr_op: None,
        }
    }
}

/// Error variants of [`PytketDecodeError`], signalling errors during the
/// conversion between tket2 ops and pytket operations.
#[derive(derive_more::Debug, Display, Error, Clone)]
#[non_exhaustive]
pub enum PytketDecodeErrorInner {
    /// The pytket circuit uses multi-indexed registers.
    //
    // This could be supported in the future, if there is a need for it.
    #[display("Register {register} in the circuit has multiple indices. Tket2 does not support multi-indexed registers")]
    MultiIndexedRegister {
        /// The register name.
        register: String,
    },
    /// Found an unexpected register name.
    #[display("Found an unknown qubit register name: {register}")]
    UnknownQubitRegister {
        /// The unknown register name.
        register: String,
    },
    /// Found an unexpected bit register name.
    #[display("Found an unknown bit register name: {register}")]
    UnknownBitRegister {
        /// The unknown register name.
        register: String,
    },
    /// The given signature to use for the HUGR's input wires is not compatible with the number of qubits and bits in the pytket circuit.
    ///
    /// The expected number of qubits and bits may be different depending on the [`PytketTypeTranslator`][extension::PytketTypeTranslator]s used in the decoder config.
    #[display(
        "The given input types {input_types} to use for the HUGR's input wires are not compatible with the number of qubits and bits in the pytket circuit. Expected {expected_qubits} qubits and {expected_bits} bits, but found {circ_qubits} qubits and {circ_bits} bits",
        input_types = input_types.iter().join(", "),
    )]
    InvalidInputSignature {
        /// The given input types.
        input_types: Vec<String>,
        /// The expected number of qubits in the signature.
        expected_qubits: usize,
        /// The expected number of bits in the signature.
        expected_bits: usize,
        /// The number of qubits in the pytket circuit.
        circ_qubits: usize,
        /// The number of bits in the pytket circuit.
        circ_bits: usize,
    },
    /// The signature to use for the HUGR's output wires is not compatible with the number of qubits and bits in the pytket circuit.
    ///
    /// We don't do any kind of type conversion, so this depends solely on the last operation to update each register.
    #[display(
        "The expected output types {expected_types} are not compatible with the actual output types {actual_types}, obtained from decoding the pytket circuit",
        expected_types = expected_types.iter().join(", "),
        actual_types = actual_types.iter().join(", "),
    )]
    InvalidOutputSignature {
        /// The expected types of the input wires.
        expected_types: Vec<String>,
        /// The actual types of the input wires.
        actual_types: Vec<String>,
    },
    /// A pytket operation had some input registers that couldn't be mapped to hugr wires.
    //
    // Some of this errors will be avoided in the future once we are able to decompose complex types automatically.
    #[display(
        "Could not find a wire with the required qubit arguments [{qubit_args:?}] and bit arguments [{bit_args:?}]",
        qubit_args = qubit_args.iter().join(", "),
        bit_args = bit_args.iter().join(", "),
    )]
    ArgumentCouldNotBeMapped {
        /// The qubit arguments that couldn't be mapped.
        qubit_args: Vec<String>,
        /// The bit arguments that couldn't be mapped.
        bit_args: Vec<String>,
    },
    /// Found an unexpected number of input wires when decoding an operation.
    #[display(
            "Expected {expected_values} input value wires{expected_types} and {expected_params} input parameters, but found {actual_values} values{actual_types} and {actual_params} parameters",
            expected_types = match expected_types {
                None => "".to_string(),
                Some(tys) => format!(" with types [{}]", tys.iter().join(", ")),
            },
            actual_types = match actual_types {
                None => "".to_string(),
                Some(tys) => format!(" with types [{}]", tys.iter().join(", ")),
            },
        )]
    UnexpectedInputWires {
        /// The expected amount of input wires.
        expected_values: usize,
        /// The expected amount of input parameters.
        expected_params: usize,
        /// The actual amount of input wires.
        actual_values: usize,
        /// The actual amount of input parameters.
        actual_params: usize,
        /// The expected types of the input wires.
        expected_types: Option<Vec<String>>,
        /// The actual types of the input wires.
        actual_types: Option<Vec<String>>,
    },
    /// Found an unexpected input type when decoding an operation.
    #[display(
        "Found an unexpected type {unknown_type} in the input wires, in input signature ({all_types})",
        all_types = all_types.iter().join(", "),
    )]
    UnexpectedInputType {
        /// The unknown type.
        unknown_type: String,
        /// All the input types specified for the operation.
        all_types: Vec<String>,
    },
    /// Tried to track the output wires of a node, but the number of tracked elements didn't match the ones in the output wires.
    #[display(
        "Tried to track the output wires of a node, but the number of tracked elements didn't match the ones in the output wires. Expected {expected_qubits} qubits and {expected_bits} bits, but found {circ_qubits} qubits and {circ_bits} bits in the node outputs"
    )]
    UnexpectedNodeOutput {
        /// The expected number of qubits.
        expected_qubits: usize,
        /// The expected number of bits.
        expected_bits: usize,
        /// The number of qubits in HUGR node outputs.
        circ_qubits: usize,
        /// The number of bits in HUGR node output.
        circ_bits: usize,
    },
    /// Custom user-defined error raised while encoding an operation.
    #[display("Error while decoding operation: {msg}")]
    CustomError {
        /// The custom error message
        msg: String,
    },
    /// Input parameter was defined multiple times.
    #[display("Parameter {param} was defined multiple times in the input signature")]
    DuplicatedParameter {
        /// The parameter name.
        param: String,
    },
    /// Not enough parameter names given for the input signature.
    #[display("Tried to initialize a pytket circuit decoder with {num_params_given} given parameter names, but more were required by the input signature")]
    MissingParametersInInput {
        /// The number of parameters given.
        num_params_given: usize,
    },
    /// We don't support complex types containing parameters in the input.
    //
    // This restriction may be relaxed in the future.
    #[display("Complex type {ty} contains {num_params} inside it. We only support input parameters in standalone 'float' or 'rotation'-typed wires")]
    UnsupportedParametersInInput {
        /// The type that contains the parameters.
        ty: String,
        /// The number of parameters in the type.
        num_params: usize,
    },
    /// We couldn't find a wire that contains the required type.
    #[display(
        "Could not find a wire with type {ty} that contains {expected_arguments}",
        expected_arguments = match (qubit_args.is_empty(), bit_args.is_empty()) {
            (true, true) => "no arguments".to_string(),
            (true, false) => format!("pytket bit arguments [{}]", bit_args.iter().join(", ")),
            (false, true) => format!("pytket qubit arguments [{}]", qubit_args.iter().join(", ")),
            (false, false) => format!("pytket qubit and bit arguments [{}] and [{}]", qubit_args.iter().join(", "), bit_args.iter().join(", ")),
        },
    )]
    NoMatchingWire {
        /// The type that couldn't be found.
        ty: String,
        /// The qubit registers expected in the wire.
        qubit_args: Vec<String>,
        /// The bit registers expected in the wire.
        bit_args: Vec<String>,
    },
    /// The number of pytket registers expected for an operation is not enough.
    ///
    /// This is usually caused by a mismatch between the input signature and the number of registers in the pytket circuit.
    ///
    /// The expected number of registers may be different depending on the [`PytketTypeTranslator`][extension::PytketTypeTranslator]s used in the decoder config.
    #[display(
        "Expected {expected_count} to map types ({expected_types}), but only got {actual_count}",
        expected_types = expected_types.iter().join(", "),
    )]
    NotEnoughPytketRegisters {
        /// The types we tried to get wires for.
        expected_types: Vec<String>,
        /// The number of registers required by the types.
        expected_count: RegisterCount,
        /// The number of registers we actually got.
        actual_count: RegisterCount,
    },
    /// A qubit was marked as outdated, but was expected to be fresh.
    #[display("Discarded qubit {qubit} cannot be used as an input")]
    OutdatedQubit {
        /// The qubit that was marked as outdated.
        qubit: String,
    },
    /// A bit was marked as outdated, but was expected to be fresh.
    #[display("Discarded bit {bit} cannot be used as an input")]
    OutdatedBit {
        /// The bit that was marked as outdated.
        bit: String,
    },
}

impl PytketDecodeErrorInner {
    /// Wrap the error in a [`PytketDecodeError`].
    pub fn wrap(self) -> PytketDecodeError {
        PytketDecodeError::from(self)
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
