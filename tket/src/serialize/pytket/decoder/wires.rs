//! Structures to keep track of pytket [`ElementId`][tket_json_rs::register::ElementId]s and
//! their correspondence to wires in the hugr being defined.

use std::sync::Arc;

use hugr::builder::{Dataflow as _, FunctionBuilder};
use hugr::extension::prelude::{bool_t, qb_t};
use hugr::ops::Value;
use hugr::std_extensions::arithmetic::float_types::ConstF64;
use hugr::types::Type;
use hugr::{Hugr, Wire};
use indexmap::IndexMap;
use itertools::Itertools;
use tket_json_rs::register;

use crate::extension::rotation::rotation_type;
use crate::serialize::pytket::decoder::param::parser::{parse_pytket_param, PytketParam};
use crate::serialize::pytket::decoder::{LoadedParameter, Tk1DecoderContext};
use crate::serialize::pytket::Tk1DecodeError;
use crate::symbolic_constant_op;

/// An identifier for a pytket register in the data carried by a wire.
///
/// After a pytket circuit assigns a new value to the register, older references
/// to it become
#[derive(Debug, Clone, PartialEq)]
pub enum TrackedElement {
    /// A tracked pytket qubit.
    Qubit(Arc<register::ElementId>),
    /// A tracked pytket classical bit.
    Bit(Arc<register::ElementId>),
    /// An outdated value for a tracked pytket qubit.
    OutdatedQubit(Arc<register::ElementId>),
    /// An outdated value for a tracked pytket classical bit.
    OutdatedBit(Arc<register::ElementId>),
}

impl TrackedElement {
    /// Returns the element id for this tracked element.
    pub fn element_id(&self) -> &register::ElementId {
        match self {
            TrackedElement::Qubit(id) | TrackedElement::OutdatedQubit(id) => id,
            TrackedElement::Bit(id) | TrackedElement::OutdatedBit(id) => id,
        }
    }

    /// Returns the type of the element.
    pub fn ty(&self) -> Type {
        match self {
            TrackedElement::Qubit(_) => qb_t(),
            TrackedElement::Bit(_) => bool_t(),
            TrackedElement::OutdatedQubit(_) => qb_t(),
            TrackedElement::OutdatedBit(_) => bool_t(),
        }
    }

    /// Returns `true` if the element has been overwritten by a new value.
    pub fn is_outdated(&self) -> bool {
        matches!(
            self,
            TrackedElement::OutdatedQubit(_) | TrackedElement::OutdatedBit(_)
        )
    }
}

/// Internal identifier for a [`TrackedElement`].
type TrackedElementId = usize;

/// Tracked data for a wire in [`InputWires`]
#[derive(Debug, Clone, PartialEq)]
pub struct WireData {
    /// The identifier in the hugr.
    wire: Wire,
    /// The type of the wire.
    ty: Arc<Type>,
    /// List of pytket arguments corresponding to this wire.
    args: Vec<TrackedElement>,
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

    /// The pytket arguments corresponding to this wire.
    pub fn args(&self) -> &[TrackedElement] {
        &self.args
    }
}

/// Input wires to a pytket operation.
#[derive(Debug, Clone)]
pub struct InputWires {
    /// Computed list of wires corresponding to the arguments,
    /// along with their types.
    wires: Vec<WireData>,
}

impl InputWires {
    /// Retrieve the wire data at the given index.
    ///
    /// Panics if the index is out of bounds. See [`InputWires::len`].
    pub fn wire_data(&self, idx: usize) -> &WireData {
        self.wires.get(idx).unwrap_or_else(|| {
            panic!(
                "Cannot get wire data at index {idx}, only {} wires are tracked",
                self.wires.len()
            )
        })
    }

    /// Return the number of wires tracked.
    ///
    /// To convert the wires into specific types and pack/unpack tuples,
    /// use [`InputWires::into_types`].
    pub fn len(&self) -> usize {
        self.wires.len()
    }

    /// Return whether there are no tracked wires.
    pub fn is_empty(&self) -> bool {
        self.wires.is_empty()
    }

    /// Return an iterator over the wires and their types.
    ///
    /// This returns the wires as-is, without any additional conversions.
    /// If you need to retrieve a specific wire type, use TODO
    pub fn iter(&self) -> impl Iterator<Item = &'_ WireData> + '_ {
        self.wires.iter()
    }

    /// Transform the current [`InputWires`] into a new `InputWires` with the given
    /// wire types, if possible.
    ///
    /// This transformation packs/unpacks tuples, converts between bool types, etc.
    ///
    /// Any wires not specified by `new_types` will be left unchanged.
    ///
    /// The `operation` parameter is a user-friendly location name used when reporting errors.
    pub fn into_types<'op>(
        self,
        new_types: impl IntoIterator<Item = &'op Type>,
        operation: &str,
        decoder: &mut Tk1DecoderContext<'_>,
    ) -> Result<InputWires, Tk1DecodeError> {
        let new_wires = WireTracker::transform_wires(self.wires, new_types, operation, decoder)?;
        Ok(InputWires { wires: new_wires })
    }

    /// Transform the current wires into a new set of wires with the given
    /// types, if possible, and return them as an array.
    ///
    /// This transformation packs/unpacks tuples, converts between bool types, etc.
    ///
    /// Any wires not specified by `new_types` will be left unchanged.
    pub fn into_types_array<'op, const N: usize>(
        self,
        new_types: &[Type; N],
        operation: &str,
        decoder: &mut Tk1DecoderContext<'_>,
    ) -> Result<([WireData; N], InputWires), Tk1DecodeError> {
        let new_wires = self.into_types(new_types, operation, decoder)?;
        let wire_arr: [WireData; N] = new_wires.wires[..N]
            .iter()
            .cloned()
            .collect_array()
            .unwrap_or_else(|| {
                panic!("Expected at least {N} wires, got {}", new_wires.wires.len())
            });
        Ok((wire_arr, new_wires))
    }

    /// Checks that we have the expected number of wires.
    ///
    /// Returns an error otherwise.
    pub fn check_len(&self, expected: usize, operation: &str) -> Result<(), Tk1DecodeError> {
        if self.wires.len() != expected {
            let types = self.wires.iter().map(|wd| wd.ty.to_string()).collect_vec();
            Err(Tk1DecodeError::UnexpectedInputWires {
                expected,
                actual: self.wires.len(),
                types,
                operation: operation.to_string(),
            })
        } else {
            Ok(())
        }
    }
}

impl<'a> IntoIterator for InputWires {
    type Item = WireData;
    type IntoIter = std::vec::IntoIter<Self::Item>;

    fn into_iter(self) -> Self::IntoIter {
        self.wires.into_iter()
    }
}

/// Tracker for wires added to a hugr.
///
/// Keeps track of the wires added to the hugr, and the last
#[derive(Debug, Clone)]
pub(super) struct WireTracker {
    /// The list of tracked elements.
    ///
    /// We use [`TrackedElementId`] to index into this list.
    elements: Vec<TrackedElement>,
    /// A list of tracked wires, with their type and list of
    /// tracked pytket elements and arguments.
    wires: IndexMap<Wire, WireData>,
    /// For each element id, the list of wires that contain it.
    /// If there exists a wire containing only the single element,
    /// it will be the first item in the list.
    element_wires: IndexMap<TrackedElementId, Vec<Wire>>,
    /// The current [`TrackedElementId`] for pytket register names.
    register_element_id: IndexMap<Arc<register::ElementId>, TrackedElementId>,
    /// An ordered set of parameters found in operation arguments, and added as
    /// new region inputs.
    parameters: IndexMap<String, LoadedParameter>,
    /// A list of input variables added to the hugr.
    ///
    /// Ordered according to their order in the function input.
    parameter_vars: Vec<String>,
}

impl WireTracker {
    /// Returns a new WireTracker.
    pub fn new() -> Self {
        WireTracker {
            elements: Vec::new(),
            wires: IndexMap::new(),
            element_wires: IndexMap::new(),
            register_element_id: IndexMap::new(),
            parameters: IndexMap::new(),
            parameter_vars: Vec::new(),
        }
    }

    /// Closes the WireTracker.
    ///
    /// Returns:
    /// - A list of input parameter added to the hugr, in order.
    pub fn finish(self) -> Vec<String> {
        self.parameter_vars
    }

    /// Returns a new set of [InputWires] for a list of
    /// [`circuit_json::Command`][tket_json_rs::circuit_json::Command] inputs.
    ///
    /// If the input elements are nested inside some composite
    pub fn wire_inputs_for_command(
        &mut self,
        hugr: &mut FunctionBuilder<&mut Hugr>,
        mut args: &[register::ElementId],
    ) -> Result<InputWires, Tk1DecodeError> {
        // We need to return a set of wires that contain all the arguments.
        //
        // We collect this by checking the wires where each element is present,
        // and collecting them in order.
        while !args.is_empty() {
            
        }
    }

    /// Transform a list of wires into an equivalent set with the given types.
    ///
    /// This transformation packs/unpacks tuples, converts between bool types, etc.
    ///
    /// The `operation` parameter is a user-friendly location name used when reporting errors.
    pub fn transform_wires<'op>(
        wires: Vec<WireData>,
        new_types: impl IntoIterator<Item = &'op Type>,
        operation: &str,
        decoder: &mut Tk1DecoderContext<'_>,
    ) -> Result<Vec<WireData>, Tk1DecodeError> {
        // If we already have the types, we can just return the wires.
        let new_types = new_types.into_iter().collect_vec();
        if wires
            .iter()
            .zip(new_types.iter())
            .all(|(wd, new_type)| wd.ty.as_ref() == *new_type)
        {
            return Ok(wires);
        }

        let new_types = new_types.into_iter();
        let new_wires: Vec<WireData> = Vec::with_capacity(new_types.size_hint().0);
        let _ = (operation, decoder, new_wires);
        // TODO: We need to implement the different mappings here.
        // Check if we can use the memoized unpacking helper from
        // [tket2_hseries::extension::qsystem::barrier].
        todo!()
    }

    /// Returns the wire carrying a parameter.
    ///
    /// - If the parameter is a known algebraic operation, adds the required op and recurses on its inputs.
    /// - If the parameter is a constant, a constant definition is added to the Hugr.
    /// - If the parameter is a variable, adds a new `rotation` input to the region.
    /// - If the parameter is a sympy expressions, adds it as a [`SympyOpDef`][crate::extension::sympy::SympyOpDef] black box.
    ///
    /// The returned wires always have float type.
    pub fn load_parameter(&mut self, hugr: &mut FunctionBuilder<&mut Hugr>, param: String) -> Wire {
        fn process(
            hugr: &mut FunctionBuilder<&mut Hugr>,
            input_params: &mut IndexMap<String, LoadedParameter>,
            param_vars: &mut Vec<String>,
            parsed: PytketParam,
            param: &str,
        ) -> LoadedParameter {
            match parsed {
                PytketParam::Constant(half_turns) => {
                    let value: Value = ConstF64::new(half_turns).into();
                    let wire = hugr.add_load_const(value);
                    LoadedParameter::float(wire)
                }
                PytketParam::Sympy(expr) => {
                    // store string in custom op.
                    let symb_op = symbolic_constant_op(expr.to_string());
                    let wire = hugr.add_dataflow_op(symb_op, []).unwrap().out_wire(0);
                    LoadedParameter::rotation(wire)
                }
                PytketParam::InputVariable { name } => {
                    // Special case for the name "pi", inserts a `ConstRotation::PI` instead.
                    if name == "pi" {
                        let value: Value = ConstF64::new(std::f64::consts::PI).into();
                        let wire = hugr.add_load_const(value);
                        return LoadedParameter::float(wire);
                    }
                    // Look it up in the input parameters to the circuit, and add a new wire if needed.
                    *input_params.entry(name.to_string()).or_insert_with(|| {
                        param_vars.push(name.to_string());
                        let wire = hugr.add_input(rotation_type());
                        LoadedParameter::rotation(wire)
                    })
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
                    LoadedParameter::float(res.out_wire(0))
                }
            }
        }

        let parsed = parse_pytket_param(&param);
        process(
            hugr,
            &mut self.parameters,
            &mut self.parameter_vars,
            parsed,
            &param,
        )
        .as_rotation(hugr)
        .wire
    }
}
