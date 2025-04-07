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

use std::borrow::Cow;
use std::collections::{BTreeMap, HashMap, HashSet};

use hugr::core::HugrNode;
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
    Tk1EncoderConfig, METADATA_B_OUTPUT_REGISTERS, METADATA_Q_OUTPUT_REGISTERS,
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
    /// List of generated qubit register names.
    qubits: Vec<RegisterUnit>,
    /// List of generated bit register names.
    bits: Vec<RegisterUnit>,
    /// List of seen parameters.
    params: Vec<String>,
    /// Counts of used qubits/bits/parameters.
    ///
    /// If these are less than the number of entries in `qubits` and `bits`,
    /// we'll use the remaining ones when allocating new registers before
    /// generating fresh names with the `RegisterUnitGenerator`s.
    counts: RegisterCount,

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

/// The result finalizing the value tracker.
///
/// Contains the final list of qubit and bit registers, and the implicit
/// permutation of the output registers.
#[derive(Debug, Clone)]
pub struct ValueTrackerResult {
    /// The final list of qubit registers.
    pub qubits: Vec<RegisterUnit>,
    /// The final list of bit registers.
    pub bits: Vec<RegisterUnit>,
    /// The implicit permutation of the qubit registers.
    pub qubit_permutation: Vec<circuit_json::ImplicitPermutation>,
    /// The implicit permutation of the bit registers.
    pub bit_permutation: Vec<circuit_json::ImplicitPermutation>,
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
            counts: RegisterCount::default(),
            wires: BTreeMap::new(),
            output_qubits: read_metadata_json_list(circ, METADATA_Q_OUTPUT_REGISTERS),
            output_bits: read_metadata_json_list(circ, METADATA_B_OUTPUT_REGISTERS),
            qubit_reg_generator: RegisterUnitGenerator::default(),
            bit_reg_generator: RegisterUnitGenerator::default(),
        };

        // Associate each input wire with a qubit/bit/parameter value.
        tracker.qubit_reg_generator = RegisterUnitGenerator::new("q", tracker.qubits.iter());
        tracker.bit_reg_generator = RegisterUnitGenerator::new("c", tracker.bits.iter());

        // Register the circuit's inputs with the tracker.
        let inp_node = circ.input_node();
        let signature = circ.circuit_signature();
        for (port, typ) in circ.hugr().node_outputs(inp_node).zip(signature.input()) {
            let wire = Wire::new(inp_node, port);
            let Some(count) = config.type_to_pytket(typ)? else {
                return Err(Tk1ConvertError::NonSerializableInputs { typ });
            };

            let mut wire_values = Vec::with_capacity(count.total());
            for _ in 0..count.qubits {
                let qb = tracker.new_qubit();
                wire_values.push(TrackedValue::Qubit(qb));
            }
            for _ in 0..count.bits {
                let bit = tracker.new_bit();
                wire_values.push(TrackedValue::Bit(bit));
            }
            for _ in 0..count.params {
                let param = tracker.new_param();
                wire_values.push(TrackedValue::Param(param));
            }

            tracker.register_values(wire, wire_values, circ);
        }

        Ok(tracker)
    }

    /// Create a new qubit register name.
    ///
    /// Picks unused names from the `qubits` list, if available, or generates
    /// a new one with the internal [`RegisterUnitGenerator`].
    pub fn new_qubit(&mut self) -> TrackedQubit {
        if self.counts.qubits >= self.qubits.len() {
            debug_assert_eq!(self.counts.qubits, self.qubits.len());
            self.qubits.push(self.qubit_reg_generator.next());
        };
        let qubit = TrackedQubit(self.counts.qubits);
        self.counts.qubits += 1;
        qubit
    }

    /// Create a new bit register name.
    ///
    /// Picks unused names from the `bits` list, if available, or generates
    /// a new one with the internal [`RegisterUnitGenerator`].
    pub fn new_bit(&mut self) -> TrackedBit {
        if self.counts.bits >= self.bits.len() {
            debug_assert_eq!(self.counts.bits, self.bits.len());
            self.bits.push(self.bit_reg_generator.next());
        };
        let bit = TrackedBit(self.counts.bits);
        self.counts.bits += 1;
        bit
    }

    /// Create a fresh parameter variable.
    ///
    /// Picks unused names from the `params` list, if available, or generates
    /// a new one formated as `x##`.
    pub fn new_param(&mut self) -> TrackedParam {
        if self.counts.params >= self.params.len() {
            debug_assert_eq!(self.counts.params, self.params.len());
            let param = "x" + &self.counts.params.to_string();
            self.params.push(param);
        };
        let param = TrackedParam(self.counts.params);
        self.counts.params += 1;
        param
    }

    /// Associate a list of values with a wire.
    ///
    /// ### Panics
    ///
    /// If the wire is already associated with a different set of values.
    pub fn register_values(
        &mut self,
        wire: Wire<N>,
        values: Vec<TrackedValue>,
        circ: &Circuit<impl HugrView<Node = N>>,
    ) {
        let unexplored_neighbours = circ.hugr().linked_ports(wire.node(), wire.source()).count();
        let present = self.wires.insert(
            wire,
            TrackedWire {
                values,
                unexplored_neighbours,
            },
        );
        if present.is_some() {
            panic!("Wire {:?} already has values associated with it", wire);
        }
    }

    /// Returns the values associated with a wire.
    ///
    /// Marks the port connection as explored. When all ports connected to the wire
    /// are explored, the wire is removed from the tracker.
    ///
    /// To avoid this use `peek_wire_values` instead.
    ///
    /// Returns `None` if the wire is not present in the tracker.
    pub fn wire_values(&mut self, wire: Wire<N>) -> Option<Cow<'_, [TrackedValue]>> {
        let values = self.wires.get_mut(&wire)?;
        match values.unexplored_neighbours {
            1 => {
                let values = self.wires.remove(&wire).unwrap();
                Some(Cow::Owned(values.values))
            }
            _ => {
                values.unexplored_neighbours -= 1;
                Some(Cow::Borrowed(&values.values))
            }
        }
    }

    /// Returns the values associated with a wire.
    ///
    /// The wire is not marked as explored. To improve performance, make sure to call
    /// [`ValueTracker::wire_values`] once per wire connection.
    ///
    /// Returns `None` if the wire is not present in the tracker.
    pub fn peek_wire_values(&self, wire: Wire<N>) -> Option<&[TrackedValue]> {
        let values = self.wires.get(&wire)?;
        Some(&values.values)
    }

    /// Returns the qubit register associated with a qubit value.
    pub fn qubit_register(&self, qb: TrackedQubit) -> &RegisterUnit {
        &self.qubits[qb.0]
    }

    /// Returns the bit register associated with a bit value.
    pub fn bit_register(&self, bit: TrackedBit) -> &RegisterUnit {
        &self.bits[bit.0]
    }

    /// Returns the string-encoded parameter associated with a parameter value.
    pub fn param_name(&self, param: TrackedParam) -> &str {
        &self.params[param.0]
    }

    /// Finish the tracker and return the final list of qubit and bit registers.
    ///
    /// Looks at the circuit's output node to determine the final order of output.
    pub fn finish(self, circ: &Circuit<impl HugrView<Node = N>>) -> ValueTrackerResult {
        // Ordered list of qubits and bits at the output of the circuit.
        let mut qubit_outputs = Vec::with_capacity(self.counts.qubits);
        let mut bit_outputs = Vec::with_capacity(self.counts.bits);
        for (node, port) in circ.hugr().all_linked_outputs(circ.output_node()) {
            let wire = Wire::new(node, port);
            if let Some(values) = self.peek_wire_values(wire) {
                for value in values {
                    match value {
                        TrackedValue::Qubit(qb) => {
                            qubit_outputs.push(self.qubit_register(*qb).clone())
                        }
                        TrackedValue::Bit(bit) => bit_outputs.push(self.bit_register(*bit).clone()),
                        TrackedValue::Param(_) => {
                            // Parameters are not part of a pytket circuit output.
                            // We ignore them here.
                        }
                    }
                }
            }
        }

        // Compute the final register permutations.
        let used_qubits = &self.qubits[..self.counts.qubits];
        let (qubit_outputs, qubit_permutation) =
            compute_final_permutation(qubit_outputs, used_qubits, &self.output_qubits);
        let used_bits = &self.bits[..self.counts.bits];
        let (bit_outputs, bit_permutation) =
            compute_final_permutation(bit_outputs, used_bits, &self.output_bits);

        ValueTrackerResult {
            qubits: qubit_outputs,
            bits: bit_outputs,
            qubit_permutation,
            bit_permutation,
        }
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

impl RegisterCount {
    /// Returns the number of qubits, bits, and parameters associated with the wire.
    pub fn total(&self) -> usize {
        self.qubits + self.bits + self.params
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

/// Compute the final unit permutation for a circuit.
///
/// Arguments:
/// - `all_inputs`: The ordered list of registers declared in the circuit.
/// - `actual_outputs`: The final order of output registers, computed from the
///   wires at the output node of the circuit.
/// - `declared_outputs`: The list of output registers declared at the start of
///   the circuit, potentially in a different order than `declared_inputs`.
///
/// Returns:
/// - The final list of output registers, including any extra registers
///   discarded mid-circuit.
/// - The final permutation of the output registers.
pub(super) fn compute_final_permutation(
    mut actual_outputs: Vec<RegisterUnit>,
    all_inputs: &[RegisterUnit],
    declared_outputs: &[RegisterUnit],
) -> (Vec<RegisterUnit>, Vec<circuit_json::ImplicitPermutation>) {
    let mut declared_outputs: Vec<&RegisterUnit> = declared_outputs.iter().collect();
    let mut declared_outputs_hashes: HashSet<RegisterHash> =
        declared_outputs.iter().map(RegisterHash::from).collect();
    let mut actual_outputs_hashes: HashSet<RegisterHash> =
        actual_outputs.iter().map(RegisterHash::from).collect();
    let mut input_hashes: HashMap<RegisterHash, usize> = HashMap::default();
    for (i, inp) in all_inputs.iter().enumerate() {
        let hash = inp.into();
        input_hashes.insert(hash, i);
        // Fix the declared output order of registers.
        if !declared_outputs_hashes.contains(&hash) {
            declared_outputs.push(inp);
            declared_outputs_hashes.insert(hash);
        }
    }
    // Extend `actual_outputs` with extra registers seen in the circuit.
    for reg in all_inputs {
        let hash = reg.into();
        if !actual_outputs_hashes.contains(&hash) {
            actual_outputs.push(reg.clone());
            actual_outputs_hashes.insert(hash);
        }
    }

    // Compute the final permutation.
    //
    // For each element `reg` at the output of the circuit, we find its position `i` at the input,
    // and find out the pytket output register associated with that position in the `declared_outputs` list.
    let permutation = actual_outputs
        .iter()
        .map(|reg| {
            let hash = reg.into();
            let i = input_hashes.get(&hash).unwrap();
            let out = declared_outputs[*i].clone();
            circuit_json::ImplicitPermutation(
                tket_json_rs::register::Qubit { id: reg.clone() },
                tket_json_rs::register::Qubit { id: out },
            )
        })
        .collect_vec();

    (actual_outputs, permutation)
}
