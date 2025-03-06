//! Tracker for pytket values associated to wires in a hugr being encoded.
//!
//! Values can be qubits or bits (identified by a [`tket_json_rs::register::ElementId`]),
//! or a string-encoded parameter expression.
//!
//! Wires in the hugr may be associated with multiple values.
//! Qubit and bit wires map to a single register element, and float/rotation wires map to a string parameter.
//! But custom operations (e.g. arrays / sums) may map to multiple things.
//!
//! Extensions can define which elements they map to

use std::collections::{BTreeMap, HashMap, HashSet};

use hugr::core::HugrNode;
use hugr::extension::prelude::qb_t;
use hugr::{HugrView, Wire};
use itertools::Itertools;
use tket_json_rs::circuit_json;
use tket_json_rs::register::ElementId as RegisterUnit;

use crate::circuit::Circuit;
use crate::serialize::pytket::{
    RegisterHash, Tk1ConvertError, METADATA_B_REGISTERS, METADATA_INPUT_PARAMETERS,
};

use super::unit_generator::RegisterUnitGenerator;
use super::{
    Tk1EncoderConfig, Tk1EncoderContext, METADATA_B_OUTPUT_REGISTERS, METADATA_Q_OUTPUT_REGISTERS,
    METADATA_Q_REGISTERS,
};

/// A structure for tracking qubits used in the circuit being encoded.
///
/// Nb: Although `tket-json-rs` has a "Register" struct, it's actually
/// an identifier for single qubits/bits in the `Register::0` register.
/// We rename it to `RegisterUnit` here to avoid confusion.
#[derive(derive_more::Debug, Clone)]
#[debug(bounds(N: std::fmt::Debug))]
pub struct ValueTracker<N> {
    /// List of seen qubit registers.
    qubits: Vec<RegisterUnit>,
    /// List of seen bit registers.
    bits: Vec<RegisterUnit>,
    /// List of seen parameters.
    params: Vec<String>,

    /// The tracked data for a wire in the hugr.
    ///
    /// Contains an ordered list of values associated with it,
    /// and a counter of unexplored neighbours used to prune the map
    /// once the wire is fully explored.
    wires: BTreeMap<Wire<N>, TrackedWire>,

    /// A fixed order for the output qubits. This is typically used by tket1 to
    /// define implicit qubit permutations at the end of the circuit.
    ///
    /// When a circuit gets decoded from pytket, we store the order in a
    /// [`METADATA_Q_OUTPUT_REGISTERS`] metadata entry.
    output_qubits: Vec<RegisterUnit>,
    /// A fixed order for the output qubits. This is typically used by tket1 to
    /// define implicit qubit permutations at the end of the circuit.
    ///
    /// When a circuit gets decoded from pytket, we store the order in a
    /// [`METADATA_B_OUTPUT_REGISTERS`] metadata entry.
    output_bits: Vec<RegisterUnit>,

    /// A generator of new registers units to use for qubit wires.
    qubit_reg_generator: RegisterUnitGenerator,
    /// A generator of new registers units to use for bit wires.
    bit_reg_generator: RegisterUnitGenerator,
}

/// A lightweight identifier for a qubit value.
///
/// Contains an index into the `qubits` array of [`ValueTracker`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, derive_more::Display)]
#[display("qubit#{}", self.0)]
pub struct TrackedQubit(usize);

/// A lightweight identifier for a bit value.
///
/// Contains an index into the `bits` array of [`ValueTracker`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, derive_more::Display)]
#[display("bit#{}", self.0)]
pub struct TrackedBit(usize);

/// A lightweight identifier for a parameter value.
///
/// Contains an index into the `params` array of [`ValueTracker`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, derive_more::Display)]
#[display("param#{}", self.0)]
pub struct TrackedParam(usize);

/// A lightweight identifier for a qubit/bit/parameter value.
///
/// Contains an index into the corresponding value array in [`ValueTracker`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TrackedValue {
    /// A qubit value.
    ///
    /// Index into the `qubits` array of [`ValueTracker`].
    Qubit(TrackedQubit),
    /// A bit value.
    ///
    /// Index into the `bits` array of [`ValueTracker`].
    Bit(TrackedBit),
    /// A parameter value.
    ///
    /// Index into the `params` array of [`ValueTracker`].
    Param(TrackedParam),
}

/// Data associated with a tracked wire in the hugr.
#[derive(Debug, Clone)]
struct TrackedWire {
    /// The values associated with the wire.
    ///
    /// This is a list of [`TrackedValue`]s, which can be qubits, bits, or
    /// parameters.
    pub values: Vec<TrackedValue>,
    /// The number of unexplored neighbours of the wire.
    ///
    /// This is used to prune the [`ValueTracker::wires`] map once the wire is
    /// fully explored.
    pub unexplored_neighbours: usize,
}

/// The number of pytket qubits, bits, and sympy parameters corresponding to a
/// HUGR type.
///
/// Used as return value for [`Tk1Encoder::type_to_pytket`](`super::Tk1Encoder::type_to_pytket`).
#[derive(
    Clone,
    Copy,
    PartialEq,
    Eq,
    Hash,
    Debug,
    Default,
    derive_more::Display,
    derive_more::Add,
    derive_more::Sub,
    derive_more::Sum,
)]
#[display("{qubits} qubits, {bits} bits, {params} parameters")]
pub struct RegisterCount {
    pub qubits: usize,
    pub bits: usize,
    pub params: usize,
}

impl<N: HugrNode> ValueTracker<N> {
    /// Create a new [`ValueTracker`] from the inputs of a [`Circuit`].
    ///
    /// Reads a number of metadata values from the circuit root node, if present, to preserve information on circuits produced by
    /// decoding a pytket circuit:
    ///
    /// - `METADATA_Q_REGISTERS`: The qubit input register names.
    /// - `METADATA_Q_OUTPUT_REGISTERS`: The reordered qubit output register names.
    /// - `METADATA_B_REGISTERS`: The bit input register names.
    /// - `METADATA_B_OUTPUT_REGISTERS`: The reordered bit output register names.
    /// - `METADATA_INPUT_PARAMETERS`: The input parameter names.
    ///
    pub fn new<H: HugrView<Node = N>>(
        circ: &Circuit<H>,
        config: Tk1EncoderConfig<H>,
    ) -> Result<Self, Tk1ConvertError> {
        let mut tracker = ValueTracker {
            qubits: read_metadata_json_list(circ, METADATA_Q_REGISTERS),
            bits: read_metadata_json_list(circ, METADATA_B_REGISTERS),
            params: read_metadata_json_list(circ, METADATA_INPUT_PARAMETERS),
            wires: BTreeMap::new(),
            output_qubits: read_metadata_json_list(circ, METADATA_Q_OUTPUT_REGISTERS),
            output_bits: read_metadata_json_list(circ, METADATA_B_OUTPUT_REGISTERS),
            qubit_reg_generator: RegisterUnitGenerator::default(),
            bit_reg_generator: RegisterUnitGenerator::default(),
        };

        // Associate each input wire with a qubit/bit/parameter value.
        tracker.qubit_reg_generator = RegisterUnitGenerator::new("q", tracker.qubits.iter());
        tracker.bit_reg_generator = RegisterUnitGenerator::new("c", tracker.bits.iter());

        // TODO: Scan the circuit input, register the wires and try to link them to `qbits`/`bits`/`params`.
        // Do something with the leftover ones.
        let mut input_counts = RegisterCount::default();
        let inp_node = circ.input_node();
        let signature = circ.circuit_signature();
        for (port, typ) in circ.hugr().node_outputs(inp_node).zip(signature.input()) {
            let wire = Wire::new(inp_node, port);
            let Some(count) = config.type_to_pytket(typ)? else {
                return Err(Tk1ConvertError::NonSerializableInputs { typ });
            };

            todo!("do something with count")
        }

        tracker
    }
}

impl TrackedWire {
    /// Returns the number of qubits, bits, and parameters associated with the wire.
    pub fn count(&self) -> RegisterCount {
        let mut count = RegisterCount::default();
        for value in &self.values {
            match value {
                TrackedValue::Qubit(_) => count.qubits += 1,
                TrackedValue::Bit(_) => count.bits += 1,
                TrackedValue::Param(_) => count.params += 1,
            }
        }
        count
    }
}

/// Read a json-encoded vector of values from the circuit's root metadata.
fn read_metadata_json_list<T: serde::de::DeserializeOwned>(
    circ: &Circuit<impl HugrView>,
    metadata_key: &str,
) -> Vec<T> {
    let Some(value) = circ.hugr().get_metadata(circ.parent(), metadata_key) else {
        return vec![];
    };

    match serde_json::from_value::<Vec<T>>(value.clone()) {
        Ok(registers) => registers,
        Err(_) => vec![],
    }
}
