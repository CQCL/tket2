//! Structures to keep track of pytket [`ElementId`][tket_json_rs::register::ElementId]s and
//! their correspondence to wires in the hugr being defined.

use std::collections::VecDeque;
use std::sync::Arc;

use hugr::builder::{Dataflow as _, FunctionBuilder};
use hugr::ops::Value;
use hugr::std_extensions::arithmetic::float_types::ConstF64;
use hugr::types::Type;
use hugr::{Hugr, Wire};
use indexmap::{IndexMap, IndexSet};
use itertools::Itertools;
use tket_json_rs::register::ElementId as PytketRegister;

use crate::extension::rotation::rotation_type;
use crate::serialize::pytket::decoder::param::parser::{parse_pytket_param, PytketParam};
use crate::serialize::pytket::decoder::{
    LoadedParameter, TrackedBit, TrackedBitId, TrackedQubit, TrackedQubitId,
};
use crate::serialize::pytket::extension::RegisterCount;
use crate::serialize::pytket::{RegisterHash, Tk1DecodeError};
use crate::symbolic_constant_op;

/// Tracked data for a wire in [`TrackedWires`].
#[derive(Debug, Clone, PartialEq)]
pub struct WireData {
    /// The identifier in the hugr.
    wire: Wire,
    /// The type of the wire.
    ty: Arc<Type>,
    /// List of pytket qubit arguments corresponding to this wire.
    qubits: Vec<TrackedQubitId>,
    /// List of pytket bit arguments corresponding to this wire.
    bits: Vec<TrackedBitId>,
}

impl WireData {
    /// The wire identifier.
    pub fn wire(&self) -> Wire {
        self.wire
    }

    /// The HUGR type for the wire.
    pub fn ty(&self) -> &Type {
        &self.ty
    }

    /// The HUGR type for the wire.
    pub fn ty_arc(&self) -> Arc<Type> {
        self.ty.clone()
    }

    /// The pytket qubit arguments corresponding to this wire.
    pub fn qubits<'d>(
        &'d self,
        wire_tracker: &'d WireTracker,
    ) -> impl Iterator<Item = TrackedQubit> + 'd {
        self.qubits
            .iter()
            .map(move |elem_id| wire_tracker.get_qubit(*elem_id))
            .cloned()
    }

    /// The pytket bit arguments corresponding to this wire.
    pub fn bits<'d>(
        &'d self,
        wire_tracker: &'d WireTracker,
    ) -> impl Iterator<Item = TrackedBit> + 'd {
        self.bits
            .iter()
            .map(move |elem_id| wire_tracker.get_bit(*elem_id))
            .cloned()
    }

    /// Returns the number of qubits carried by this wire.
    pub fn num_qubits(&self) -> usize {
        self.qubits.len()
    }

    /// Returns the number of bits carried by this wire.
    pub fn num_bits(&self) -> usize {
        self.bits.len()
    }

    /// Returns the number of tracked elements in this wire.
    pub fn num_args(&self) -> usize {
        self.num_qubits() + self.num_bits()
    }
}

/// Tracked wires to a pytket operation.
#[derive(Debug, Clone)]
pub struct TrackedWires {
    /// Computed list of wires corresponding to the arguments,
    /// along with their types.
    value_wires: Vec<WireData>,
    /// List of wires corresponding to the parameters.
    parameter_wires: Vec<Arc<LoadedParameter>>,
}

impl TrackedWires {
    /// Retrieve the wire data at the given index.
    ///
    /// Panics if the index is out of bounds. See [`TrackedWires::len`].
    #[inline]
    #[must_use]
    pub fn value_wire(&self, idx: usize) -> &WireData {
        self.value_wires.get(idx).unwrap_or_else(|| {
            panic!(
                "Cannot get wire data at index {idx}, only {} wires are tracked",
                self.value_wires.len()
            )
        })
    }

    /// Return the number of value wires tracked.
    #[inline]
    #[must_use]
    pub fn value_count(&self) -> usize {
        self.value_wires.len()
    }

    /// Return the number of parameter wires tracked.
    #[inline]
    #[must_use]
    pub fn parameter_count(&self) -> usize {
        self.parameter_wires.len()
    }

    /// Return the number of wires tracked.
    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        self.value_wires.len() + self.parameter_wires.len()
    }

    /// Return whether there are no tracked wires.
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.value_wires.is_empty() && self.parameter_wires.is_empty()
    }

    /// Return an iterator over the wires and their types.
    ///
    /// This returns the wires as-is, without any additional conversions.
    /// If you need to retrieve a specific wire type, use TODO
    #[inline]
    pub fn iter_values(&self) -> impl Iterator<Item = &'_ WireData> + Clone + '_ {
        self.value_wires.iter()
    }

    /// Return an iterator over the parameters.
    #[inline]
    pub fn iter_parameters(&self) -> impl Iterator<Item = &'_ LoadedParameter> + Clone + '_ {
        self.parameter_wires.iter().map(|p| p.as_ref())
    }

    /// Returns the types of the value wires.
    #[inline]
    pub fn value_types(&self) -> impl Iterator<Item = &'_ Type> + Clone + '_ {
        self.value_wires.iter().map(|wd| wd.ty())
    }

    /// Returns the types of the parameter wires.
    #[inline]
    pub fn parameter_types(&self) -> impl Iterator<Item = &'_ Type> + Clone + '_ {
        self.parameter_wires.iter().map(|p| p.wire_type())
    }

    /// Returns the wire types in this tracked wires.
    #[inline]
    pub fn wire_types(&self) -> impl Iterator<Item = &'_ Type> + Clone + '_ {
        self.value_types().chain(self.parameter_types())
    }

    /// Returns the tracked qubit elements in the set of wires.
    #[inline]
    pub fn qubits<'d>(
        &'d self,
        wire_tracker: &'d WireTracker,
    ) -> impl Iterator<Item = TrackedQubit> + 'd {
        self.value_wires
            .iter()
            .flat_map(move |wd| wd.qubits(wire_tracker))
    }

    /// Returns the tracked qubit elements in the set of wires as an array.
    #[inline]
    pub fn qubits_arr<const N: usize>(
        &self,
        wire_tracker: &WireTracker,
    ) -> Option<[TrackedQubit; N]> {
        self.qubits(wire_tracker).collect_array()
    }

    /// Returns the tracked bit elements in the set of wires.
    #[inline]
    pub fn bits<'d>(
        &'d self,
        wire_tracker: &'d WireTracker,
    ) -> impl Iterator<Item = TrackedBit> + 'd {
        self.value_wires
            .iter()
            .flat_map(move |wd| wd.bits(wire_tracker))
    }

    /// Return the tracked value wires in this tracked wires.
    #[inline]
    pub fn value_wires(&self) -> impl Iterator<Item = Wire> + Clone + '_ {
        self.value_wires.iter().map(|wd| wd.wire())
    }

    /// Return the tracked parameter wires in this tracked wires.
    #[inline]
    pub fn parameter_wires(&self) -> impl Iterator<Item = Wire> + Clone + '_ {
        self.parameter_wires.iter().map(|p| p.wire)
    }

    /// Returns the wires in this tracked wires.
    #[inline]
    pub fn wires(&self) -> impl Iterator<Item = Wire> + Clone + '_ {
        self.value_wires().chain(self.parameter_wires())
    }

    /// Returns the wires in this tracked wires as an array of types.
    ///
    /// Returns an error if the number of wires is not equal to `N`.
    ///
    /// # Arguments
    ///
    /// * `operation` - The name of the operation being decoded, used for error reporting.
    #[inline]
    pub fn wires_arr<const N: usize>(&self, operation: &str) -> Result<[Wire; N], Tk1DecodeError> {
        let expected_values = N.saturating_sub(self.parameter_count());
        let expected_params = N - expected_values;
        self.check_len(expected_values, expected_params, operation)?;
        Ok(self
            .wires()
            .collect_array()
            .expect("check_len should have failed"))
    }

    /// Returns the amount of qubits, bits, and parameters carried by this tracked wires.
    #[inline]
    #[must_use]
    pub fn register_count(&self) -> RegisterCount {
        let mut counts: RegisterCount = self
            .iter_values()
            .map(|w| RegisterCount::new(w.num_qubits(), w.num_bits(), 0))
            .sum();
        counts.params += self.parameter_count();
        counts
    }

    /// Checks that we have the expected number of wires, and returns an error otherwise.
    ///
    /// # Arguments
    ///
    /// * `expected_values` - The expected number of value wires.
    /// * `expected_params` - The expected number of parameter wires.
    /// * `operation` - The name of the operation being decoded, used for error reporting.
    pub fn check_len(
        &self,
        expected_values: usize,
        expected_params: usize,
        operation: &str,
    ) -> Result<(), Tk1DecodeError> {
        if self.value_count() != expected_values || self.parameter_count() != expected_params {
            let types = self.wire_types().map(|ty| ty.to_string()).collect_vec();
            Err(Tk1DecodeError::UnexpectedInputWires {
                expected_values,
                expected_params,
                actual_values: self.value_count(),
                actual_params: self.parameter_count(),
                expected_types: None,
                actual_types: Some(types),
                operation: operation.to_string(),
            })
        } else {
            Ok(())
        }
    }

    /// Checks that we have the expected wire types, and returns an error otherwise.
    ///
    /// # Arguments
    ///
    /// * `expected_values` - The expected types of the value wires.
    /// * `expected_params` - The expected number of parameters. Note that these may be either `float` or `rotation`-typed.
    ///   Use [`LoadedParameter::with_type`] to cast them as needed.
    /// * `operation` - The name of the operation being decoded, used for error reporting.
    pub fn check_types(
        &self,
        expected_values: &[Type],
        expected_params: usize,
        operation: &str,
    ) -> Result<(), Tk1DecodeError> {
        let vals = expected_values.iter();
        if !itertools::equal(self.value_types(), vals) || self.parameter_count() != expected_params
        {
            let actual = self.value_types().collect_vec();
            Err(Tk1DecodeError::UnexpectedInputWires {
                expected_values: expected_values.len(),
                expected_params,
                actual_values: self.value_count(),
                actual_params: self.parameter_count(),
                expected_types: Some(
                    expected_values
                        .iter()
                        .map(|ty| ty.to_string())
                        .collect_vec(),
                ),
                actual_types: Some(actual.iter().map(|ty| ty.to_string()).collect_vec()),
                operation: operation.to_string(),
            })
        } else {
            Ok(())
        }
    }
}

/// Tracker for wires added to a hugr.
///
/// Keeps track of the wires added to the hugr, and the qubit/bit/parameters
/// that they contain.
///
/// Wire may contain either a single [`LoadedParameter`] or a collection of
/// [`TrackedQubit`]s and [`TrackedBit`]s. Each tracked
/// element in a wire is said to be "up to date" if it is the latest reference
/// to that pytket register. Once the register is seen in the output of an
/// operation, all previous references to it become "outdated".
#[derive(Debug, Clone, Default)]
pub struct WireTracker {
    /// A list of tracked wires, with their type and list of
    /// tracked pytket elements and arguments.
    wires: IndexMap<Wire, WireData>,
    /// The list of tracked qubit elements.
    ///
    /// Indexed by [`TrackedQubitId`].
    qubits: Vec<TrackedQubit>,
    /// The list of tracked bit elements.
    ///
    /// Indexed by [`TrackedBitId`].
    bits: Vec<TrackedBit>,
    /// A map from pytket register hashes to the latest up-to-date [`TrackedQubit`] referencing it.
    ///
    /// The map keys are kept in the order they were defined in the circuit.
    latest_qubit_tracker: IndexMap<RegisterHash, TrackedQubitId>,
    /// A map from pytket register hashes to the latest up-to-date [`TrackedBit`] referencing it.
    ///
    /// The map keys are kept in the order they were defined in the circuit.
    latest_bit_tracker: IndexMap<RegisterHash, TrackedBitId>,
    /// For each tracked qubit, the list of wires that contain it.
    qubit_wires: IndexMap<TrackedQubitId, Vec<Wire>>,
    /// For each tracked bit, the list of wires that contain it.
    bit_wires: IndexMap<TrackedBitId, Vec<Wire>>,
    /// An ordered set of parameters found in operation arguments, and added as
    /// new region inputs.
    parameters: IndexMap<String, Arc<LoadedParameter>>,
    /// A list of input variables added to the hugr.
    ///
    /// Ordered according to their order in the function input.
    parameter_vars: IndexSet<String>,
}

impl WireTracker {
    /// Returns a new WireTracker with the given capacity.
    pub fn with_capacity(qubit_count: usize, bit_count: usize) -> Self {
        WireTracker {
            wires: IndexMap::new(),
            qubits: Vec::with_capacity(qubit_count),
            bits: Vec::with_capacity(bit_count),
            latest_qubit_tracker: IndexMap::with_capacity(qubit_count),
            latest_bit_tracker: IndexMap::with_capacity(bit_count),
            qubit_wires: IndexMap::with_capacity(qubit_count),
            bit_wires: IndexMap::with_capacity(bit_count),
            parameters: IndexMap::new(),
            parameter_vars: IndexSet::new(),
        }
    }

    /// Closes the WireTracker.
    ///
    /// Returns:
    /// - A list of qubit and bit elements, in the order they were added.
    /// - A list of input parameter added to the hugr, in the order they were added.
    pub(super) fn finish(self) -> IndexSet<String> {
        self.parameter_vars
    }

    /// Returns a reference to the tracked qubit at the given index.
    fn get_qubit(&self, id: TrackedQubitId) -> &TrackedQubit {
        &self.qubits[id.0]
    }

    /// Returns a reference to the tracked bit at the given index.
    fn get_bit(&self, id: TrackedBitId) -> &TrackedBit {
        &self.bits[id.0]
    }

    /// Returns `true` if the given register is a known bit register.
    pub(super) fn is_known_bit(&self, register: &PytketRegister) -> bool {
        self.latest_bit_tracker
            .contains_key(&RegisterHash::from(register))
    }

    /// Returns the list of known pytket registers, in the order they were registered.
    pub(super) fn known_pytket_qubits(&self) -> impl Iterator<Item = &TrackedQubit> {
        self.latest_qubit_tracker
            .iter()
            .map(|(_, &elem_id)| self.get_qubit(elem_id))
    }

    /// Returns the list of known pytket bit registers, in the order they were registered.
    pub(super) fn known_pytket_bits(&self) -> impl Iterator<Item = &TrackedBit> {
        self.latest_bit_tracker
            .iter()
            .map(|(_, &elem_id)| self.get_bit(elem_id))
    }

    /// Track a new pytket qubit register.
    ///
    /// If the pytket register was already in the tracker,
    /// marks the previous element as outdated.
    ///
    /// Returns the hash of the register.
    pub(super) fn track_qubit(
        &mut self,
        qubit_reg: Arc<PytketRegister>,
    ) -> Result<TrackedQubitId, Tk1DecodeError> {
        check_register(&qubit_reg)?;

        let id = TrackedQubitId(self.qubits.len());
        let hash = RegisterHash::from(qubit_reg.as_ref());
        self.qubits.push(TrackedQubit::new(qubit_reg));
        if let Some(previous_id) = self.latest_qubit_tracker.insert(hash, id) {
            self.qubits[previous_id.0].mark_outdated();
        }
        Ok(id)
    }

    /// Track a new pytket bit register.
    ///
    /// If the pytket register was already in the tracker,
    /// marks the previous element as outdated.
    ///
    /// Returns the hash of the register.
    pub(super) fn track_bit(
        &mut self,
        bit_reg: Arc<PytketRegister>,
    ) -> Result<TrackedBitId, Tk1DecodeError> {
        check_register(&bit_reg)?;

        let id = TrackedBitId(self.bits.len());
        let hash = RegisterHash::from(bit_reg.as_ref());
        self.bits.push(TrackedBit::new(bit_reg));
        if let Some(previous_id) = self.latest_bit_tracker.insert(hash, id) {
            self.bits[previous_id.0].mark_outdated();
        }

        Ok(id)
    }

    /// Returns the latest tracked qubit for a pytket register.
    ///
    /// Returns an error if the register is not known.
    ///
    /// The returned element is guaranteed to be up to date (See [`TrackedQubit::is_outdated`]).
    pub fn tracked_qubit_for_register(
        &self,
        register: impl AsRef<PytketRegister>,
    ) -> Result<&TrackedQubit, Tk1DecodeError> {
        let hash = RegisterHash::from(register.as_ref());
        let Some(id) = self.latest_qubit_tracker.get(&hash) else {
            return Err(Tk1DecodeError::unknown_qubit_reg(register.as_ref()));
        };
        Ok(self.get_qubit(*id))
    }

    /// Returns the latest tracked bit for a pytket register.
    ///
    /// Returns an error if the register is not known.
    ///
    /// The returned element is guaranteed to be up to date (See [`TrackedBit::is_outdated`]).
    pub fn tracked_bit_for_register(
        &self,
        register: impl AsRef<PytketRegister>,
    ) -> Result<&TrackedBit, Tk1DecodeError> {
        let hash = RegisterHash::from(register.as_ref());
        let Some(id) = self.latest_bit_tracker.get(&hash) else {
            return Err(Tk1DecodeError::unknown_bit_reg(register.as_ref()));
        };
        Ok(self.get_bit(*id))
    }

    /// Returns a new set of [TrackedWires] for a list of
    /// [`circuit_json::Command`][tket_json_rs::circuit_json::Command] inputs.
    ///
    /// Returns an error if a valid set cannot be found.
    ///
    /// # Arguments
    ///
    /// * `hugr` - The hugr to add the wires to.
    /// * `args` - The list of pytket element ids to map to wires.
    /// * `operation` - The name of the operation being decoded, used for error reporting.
    /// * `params` - The list of parameters to load to wires. See [`WireTracker::load_parameter`] for more details.
    ///
    // TODO: We'll need to be able to decompose types when we need only _some_
    // of the elements they contain (E.g., extract a value from an array),
    // and do it automatically here.
    pub(super) fn wire_inputs_for_command<'r>(
        &mut self,
        hugr: &mut FunctionBuilder<&mut Hugr>,
        qubit_args: impl IntoIterator<Item = &'r PytketRegister>,
        bit_args: impl IntoIterator<Item = &'r PytketRegister>,
        params: impl IntoIterator<Item = &'r str>,
        operation: &str,
    ) -> Result<TrackedWires, Tk1DecodeError> {
        // We need to return a set of wires that contain all the arguments.
        //
        // We collect this by checking the wires where each element is present,
        // and collecting them in order.
        let mut qubit_args: VecDeque<(TrackedQubitId, &PytketRegister)> = qubit_args
            .into_iter()
            .map(
                |register| match self.latest_qubit_tracker.get(&RegisterHash::from(register)) {
                    Some(id) => Ok((*id, register)),
                    None => Err(Tk1DecodeError::unknown_qubit_reg(register)),
                },
            )
            .collect::<Result<_, _>>()?;
        let mut bit_args: VecDeque<(TrackedBitId, &PytketRegister)> = bit_args
            .into_iter()
            .map(
                |register| match self.latest_bit_tracker.get(&RegisterHash::from(register)) {
                    Some(id) => Ok((*id, register)),
                    None => Err(Tk1DecodeError::unknown_bit_reg(register)),
                },
            )
            .collect::<Result<_, _>>()?;

        let mut value_wires = Vec::new();
        while !qubit_args.is_empty() || !bit_args.is_empty() {
            // Check candidate wires that only contain the elements we need, in the right order.
            let filter_candidate_wire = |w: Wire| {
                let mut wire_qubits = self.wires[&w].qubits.iter().peekable();
                let mut wire_bits = self.wires[&w].bits.iter().peekable();
                let mut q_args_iter = qubit_args.iter().map(|(id, _)| id);
                let mut b_args_iter = bit_args.iter().map(|(id, _)| id);

                // Check that each argument appears as either a qubit or a bit
                // in the wire, in the right order.
                //
                // We may have leftover arguments at the end, which we'll try to
                // get from another wire.
                while wire_qubits.peek().is_some() && wire_bits.peek().is_some() {
                    if let Some(qb) = wire_qubits.next() {
                        match q_args_iter.next() {
                            Some(arg) if qb == arg => continue,
                            _ => return false,
                        };
                    }
                    if let Some(bit) = wire_bits.next() {
                        match b_args_iter.next() {
                            Some(arg) if bit == arg => continue,
                            _ => return false,
                        };
                    }
                    return false;
                }
                true
            };

            let qubit_candidates = qubit_args
                .front()
                .into_iter()
                .flat_map(|(id, _)| self.qubit_wires[id].iter());
            let bit_candidates = bit_args
                .front()
                .into_iter()
                .flat_map(|(id, _)| self.bit_wires[id].iter());
            let candidate = qubit_candidates
                .chain(bit_candidates)
                .find(|&&w| filter_candidate_wire(w));

            // If we found a candidate, add it to the list of wires.
            match candidate {
                Some(w) => {
                    // Consume the extracted args, and add the wire to the list.
                    let wire_data: WireData = self.wires[w].clone();
                    qubit_args.drain(..wire_data.num_qubits());
                    bit_args.drain(..wire_data.num_bits());
                    value_wires.push(wire_data);
                }
                None => {
                    // In the future we may be able to decompose some wire containing `arg_ids[0]` internally.
                    // For now, we just report the error.
                    return Err(Tk1DecodeError::ArgumentCouldNotBeMapped {
                        operation: operation.to_string(),
                        qubit_args: qubit_args
                            .iter()
                            .map(|(_, elem)| elem.to_string())
                            .collect(),
                        bit_args: bit_args.iter().map(|(_, elem)| elem.to_string()).collect(),
                    });
                }
            }
        }

        // Load the parameters.
        let parameter_wires = params
            .into_iter()
            .map(|param| self.load_parameter(hugr, param))
            .collect_vec();

        Ok(TrackedWires {
            value_wires,
            parameter_wires,
        })
    }

    /// Returns the wire carrying a parameter.
    ///
    /// - If the parameter is a known algebraic operation, adds the required op and recurses on its inputs.
    /// - If the parameter is a constant, a constant definition is added to the Hugr.
    /// - If the parameter is a variable, adds a new `rotation` input to the region.
    /// - If the parameter is a sympy expressions, adds it as a [`SympyOpDef`][crate::extension::sympy::SympyOpDef] black box.
    pub fn load_parameter(
        &mut self,
        hugr: &mut FunctionBuilder<&mut Hugr>,
        param: &str,
    ) -> Arc<LoadedParameter> {
        fn process(
            hugr: &mut FunctionBuilder<&mut Hugr>,
            input_params: &mut IndexMap<String, Arc<LoadedParameter>>,
            param_vars: &mut IndexSet<String>,
            parsed: PytketParam,
            param: &str,
        ) -> Arc<LoadedParameter> {
            match parsed {
                PytketParam::Constant(half_turns) => {
                    let value: Value = ConstF64::new(half_turns).into();
                    let wire = hugr.add_load_const(value);
                    Arc::new(LoadedParameter::float(wire))
                }
                PytketParam::Sympy(expr) => {
                    // store string in custom op.
                    let symb_op = symbolic_constant_op(expr.to_string());
                    let wire = hugr.add_dataflow_op(symb_op, []).unwrap().out_wire(0);
                    Arc::new(LoadedParameter::rotation(wire))
                }
                PytketParam::InputVariable { name } => {
                    // Special case for the name "pi", inserts a `ConstRotation::PI` instead.
                    if name == "pi" {
                        let value: Value = ConstF64::new(std::f64::consts::PI).into();
                        let wire = hugr.add_load_const(value);
                        return Arc::new(LoadedParameter::float(wire));
                    }
                    // Look it up in the input parameters to the circuit, and add a new wire if needed.
                    input_params
                        .entry(name.to_string())
                        .or_insert_with(|| {
                            param_vars.insert(name.to_string());
                            let wire = hugr.add_input(rotation_type());
                            Arc::new(LoadedParameter::rotation(wire))
                        })
                        .clone()
                }
                PytketParam::Operation { op, args } => {
                    // We assume all operations take float inputs.
                    let input_wires = args
                        .into_iter()
                        .map(|arg| {
                            process(hugr, input_params, param_vars, arg, param)
                                .as_float(hugr)
                                .wire
                        })
                        .collect_vec();
                    // If any of the following asserts panics, it means we added invalid ops to the sympy parser.
                    let res = hugr.add_dataflow_op(op, input_wires).unwrap_or_else(|e| {
                        panic!("Error while decoding pytket operation parameter \"{param}\". {e}",)
                    });
                    assert_eq!(res.num_value_outputs(), 1, "An operation decoded from the pytket op parameter \"{param}\" had {} outputs", res.num_value_outputs());
                    Arc::new(LoadedParameter::float(res.out_wire(0)))
                }
            }
        }

        let parsed = parse_pytket_param(param);
        process(
            hugr,
            &mut self.parameters,
            &mut self.parameter_vars,
            parsed,
            param,
        )
    }

    /// Track a new wire, updating any tracked elements that are present in it.
    pub fn track_wire(
        &mut self,
        wire: Wire,
        ty: Arc<Type>,
        qubits: impl IntoIterator<Item = TrackedQubit>,
        bits: impl IntoIterator<Item = TrackedBit>,
    ) -> Result<(), Tk1DecodeError> {
        let qubits = qubits
            .into_iter()
            .map(|q| self.track_qubit(q.pytket_register_arc()))
            .collect::<Result<_, _>>()?;
        let bits = bits
            .into_iter()
            .map(|b| self.track_bit(b.pytket_register_arc()))
            .collect::<Result<_, _>>()?;

        for &q in &qubits {
            self.qubit_wires.entry(q).or_default().push(wire);
        }
        for &b in &bits {
            self.bit_wires.entry(b).or_default().push(wire);
        }

        let wire_data = WireData {
            wire,
            ty,
            qubits,
            bits,
        };
        self.wires.insert(wire, wire_data);

        Ok(())
    }

    pub(crate) fn register_input_parameter(
        &mut self,
        wire: Wire,
        param: String,
    ) -> Result<(), Tk1DecodeError> {
        let entry = self.parameters.entry(param);
        if let indexmap::map::Entry::Occupied(_) = &entry {
            return Err(Tk1DecodeError::DuplicatedParameter {
                param: entry.key().clone(),
            });
        }
        entry.insert_entry(Arc::new(LoadedParameter::rotation(wire)));
        Ok(())
    }
}

/// Only single-indexed registers are supported.
fn check_register(register: &PytketRegister) -> Result<(), Tk1DecodeError> {
    if register.1.len() != 1 {
        Err(Tk1DecodeError::MultiIndexedRegister {
            register: register.to_string(),
        })
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::serialize::pytket::decoder::Tk1DecodeError;
    use hugr::extension::prelude::{bool_t, qb_t};
    use hugr::types::SumType;
    use hugr::Node;
    use rstest::{fixture, rstest};
    use std::sync::Arc;
    use tket_json_rs::register::ElementId;

    #[fixture]
    fn sample_wire(#[default(0)] wire_idx: usize) -> Wire {
        Wire::new(Node::from(portgraph::NodeIndex::new(wire_idx)), 0)
    }

    // Test basic WireTracker creation
    #[rstest]
    fn tracker_properties() {
        let mut tracker = WireTracker::with_capacity(5, 3);
        let qubit_reg = Arc::new(ElementId("q".to_string(), vec![0]));
        let bit_reg = Arc::new(ElementId("c".to_string(), vec![0]));
        let multi_indexed_reg = Arc::new(ElementId("q".to_string(), vec![0, 1]));
        let wire1 = sample_wire(1);

        // Initially, everything is empty - test through public methods
        assert_eq!(tracker.known_pytket_qubits().count(), 0);
        assert_eq!(tracker.known_pytket_bits().count(), 0);

        // Track an invalid register name.
        match tracker.track_qubit(multi_indexed_reg.clone()) {
            Err(Tk1DecodeError::MultiIndexedRegister { register }) => {
                assert_eq!(register, multi_indexed_reg.to_string());
            }
            e => panic!("Expected MultiIndexedRegister error, got {e:?}"),
        }

        // Getting the tracked qubits or bits for an unknown register should fail.
        match tracker.tracked_qubit_for_register(&qubit_reg) {
            Err(Tk1DecodeError::UnknownQubitRegister { register }) => {
                assert_eq!(register, qubit_reg.to_string());
            }
            e => panic!("Expected UnknownQubitRegister error, got {e:?}"),
        }
        match tracker.tracked_bit_for_register(&bit_reg) {
            Err(Tk1DecodeError::UnknownBitRegister { register }) => {
                assert_eq!(register, bit_reg.to_string());
            }
            e => panic!("Expected UnknownBitRegister error, got {e:?}"),
        }

        // Track a new qubit
        let tracked_q_0 = tracker
            .track_qubit(qubit_reg.clone())
            .expect("Should track qubit");
        assert_eq!(tracker.known_pytket_qubits().count(), 1);
        assert_eq!(tracker.known_pytket_bits().count(), 0);
        let tracked_qubit = tracker
            .tracked_qubit_for_register(&qubit_reg)
            .expect("Should find tracked qubit");
        assert!(!tracked_qubit.is_outdated());
        assert_eq!(tracked_qubit, tracker.get_qubit(tracked_q_0));

        // Track the same qubit again, it should add a new TrackedQubit and mark the previous one as outdated
        let tracked_q_1 = tracker
            .track_qubit(qubit_reg.clone())
            .expect("Should track qubit again");
        assert_eq!(tracker.known_pytket_qubits().count(), 1); // still only one unique register
        assert!(tracker.get_qubit(tracked_q_0).is_outdated());
        assert!(!tracker.get_qubit(tracked_q_1).is_outdated());
        let tracked_qubit = tracker
            .tracked_qubit_for_register(&qubit_reg)
            .expect("Should find latest tracked qubit")
            .clone();
        assert_eq!(&tracked_qubit, tracker.get_qubit(tracked_q_1));

        // Track a bit
        let bit_id = tracker
            .track_bit(bit_reg.clone())
            .expect("Should track bit");
        assert_eq!(tracker.known_pytket_bits().count(), 1);
        assert!(!tracker.get_bit(bit_id).is_outdated());
        let tracked_bit = tracker
            .tracked_bit_for_register(&bit_reg)
            .expect("Should find tracked bit")
            .clone();
        assert_eq!(&tracked_bit, tracker.get_bit(bit_id));

        // Associate the bit and qubit with a wire.
        tracker
            .track_wire(
                wire1,
                Arc::new(SumType::new_tuple(vec![qb_t(), bool_t()]).into()),
                vec![tracked_qubit.clone()],
                vec![tracked_bit.clone()],
            )
            .expect("Should track wire");
    }
}
