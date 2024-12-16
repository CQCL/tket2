//! Intermediate structure for decoding [`SerialCircuit`]s into [`Hugr`]s.

use std::collections::{HashMap, HashSet};

use hugr::builder::{Container, Dataflow, DataflowHugr, FunctionBuilder};
use hugr::extension::prelude::{bool_t, qb_t};

use hugr::ops::handle::NodeHandle;
use hugr::ops::{OpType, Value};
use hugr::std_extensions::arithmetic::float_types::ConstF64;
use hugr::types::Signature;
use hugr::{Hugr, Wire};

use derive_more::Display;
use indexmap::IndexMap;
use itertools::{EitherOrBoth, Itertools};
use serde_json::json;
use tket_json_rs::circuit_json;
use tket_json_rs::circuit_json::SerialCircuit;
use tket_json_rs::register;

use super::op::Tk1Op;
use super::param::decode::{parse_pytket_param, PytketParam};
use super::{
    OpConvertError, RegisterHash, TK1ConvertError, METADATA_B_OUTPUT_REGISTERS,
    METADATA_B_REGISTERS, METADATA_OPGROUP, METADATA_PHASE, METADATA_Q_OUTPUT_REGISTERS,
    METADATA_Q_REGISTERS,
};
use crate::extension::rotation::{rotation_type, RotationOp};
use crate::extension::TKET1_EXTENSION_ID;
use crate::serialize::pytket::METADATA_INPUT_PARAMETERS;
use crate::symbolic_constant_op;

/// The state of an in-progress [`FunctionBuilder`] being built from a [`SerialCircuit`].
///
/// Mostly used to define helper internal methods.
#[derive(Debug, Clone)]
pub(super) struct Tk1Decoder {
    /// The Hugr being built.
    pub hugr: FunctionBuilder<Hugr>,
    /// A map from the tracked pytket registers to the [`Wire`]s in the circuit.
    register_wires: HashMap<RegisterHash, Wire>,
    /// The ordered list of register to have at the output.
    ordered_registers: Vec<RegisterHash>,
    /// A set of registers that encode qubits.
    qubit_registers: HashSet<RegisterHash>,
    /// An ordered set of parameters found in operation arguments, and added as inputs.
    parameters: IndexMap<String, LoadedParameter>,
}

impl Tk1Decoder {
    /// Initialize a new [`Tk1Decoder`], using the metadata from a [`SerialCircuit`].
    pub fn try_new(serialcirc: &SerialCircuit) -> Result<Self, TK1ConvertError> {
        let num_qubits = serialcirc.qubits.len();
        let num_bits = serialcirc.bits.len();
        let sig =
            Signature::new_endo([vec![qb_t(); num_qubits], vec![bool_t(); num_bits]].concat())
                .with_extension_delta(TKET1_EXTENSION_ID);

        let name = serialcirc.name.clone().unwrap_or_default();
        let mut dfg = FunctionBuilder::new(name, sig).unwrap();
        let dangling_wires = dfg.input_wires().collect::<Vec<_>>();

        // Metadata. The circuit requires "name", and we store other things that
        // should pass through the serialization roundtrip.
        dfg.set_metadata(METADATA_PHASE, json!(serialcirc.phase));
        dfg.set_metadata(METADATA_Q_REGISTERS, json!(serialcirc.qubits));
        dfg.set_metadata(METADATA_B_REGISTERS, json!(serialcirc.bits));

        // Compute the output register reordering, and store it in the metadata.
        //
        // The `implicit_permutation` field is a dictionary mapping input
        // registers to output registers on the same path.
        //
        // Here we store an ordered list showing the order in which the input
        // registers appear in the output.
        //
        // For a circuit with three qubit registers 0, 1, 2 and an implicit
        // permutation {0 -> 1, 1 -> 2, 2 -> 0}, `output_to_input` will be
        // {1 -> 0, 2 -> 1, 0 -> 2} and the output order will be [2, 0, 1].
        // That is, at position 0 of the output we'll see the register originally
        // named 2, at position 1 the register originally named 0, and so on.
        let mut output_qubits = Vec::with_capacity(serialcirc.qubits.len());
        let mut output_bits = Vec::with_capacity(serialcirc.bits.len());
        let output_to_input: HashMap<register::ElementId, register::ElementId> = serialcirc
            .implicit_permutation
            .iter()
            .map(|p| (p.1.clone().id, p.0.clone().id))
            .collect();
        for qubit in &serialcirc.qubits {
            // For each output position, find the input register that should be there.
            output_qubits.push(output_to_input.get(&qubit.id).unwrap_or(&qubit.id).clone());
        }
        for bit in &serialcirc.bits {
            // For each output position, find the input register that should be there.
            output_bits.push(output_to_input.get(&bit.id).unwrap_or(&bit.id).clone());
        }
        dfg.set_metadata(METADATA_Q_OUTPUT_REGISTERS, json!(output_qubits));
        dfg.set_metadata(METADATA_B_OUTPUT_REGISTERS, json!(output_bits));

        let qubit_registers = serialcirc.qubits.iter().map(RegisterHash::from).collect();

        let ordered_registers = serialcirc
            .qubits
            .iter()
            .map(|qb| &qb.id)
            .chain(serialcirc.bits.iter().map(|bit| &bit.id))
            .map(|reg| {
                check_register(reg)?;
                Ok(RegisterHash::from(reg))
            })
            .collect::<Result<Vec<RegisterHash>, TK1ConvertError>>()?;

        // Map each register element to their starting wire.
        let register_wires: HashMap<RegisterHash, Wire> = ordered_registers
            .iter()
            .copied()
            .zip(dangling_wires)
            .collect();

        Ok(Tk1Decoder {
            hugr: dfg,
            register_wires,
            ordered_registers,
            qubit_registers,
            parameters: IndexMap::new(),
        })
    }

    /// Finish building the [`Hugr`].
    pub fn finish(mut self) -> Hugr {
        // Order the final wires according to the serial circuit register order.
        let mut outputs = Vec::with_capacity(self.ordered_registers.len());
        for register in self.ordered_registers {
            let wire = self.register_wires.remove(&register).unwrap();
            outputs.push(wire);
        }
        debug_assert!(
            self.register_wires.is_empty(),
            "Some output wires were not associated with a register."
        );

        // Store the name for the input parameter wires
        if !self.parameters.is_empty() {
            let params = self.parameters.keys().cloned().collect_vec();
            self.hugr
                .set_metadata(METADATA_INPUT_PARAMETERS, json!(params));
        }

        self.hugr.finish_hugr_with_outputs(outputs).unwrap()
    }

    /// Add a tket1 [`circuit_json::Command`] from the serial circuit to the
    /// decoder.
    pub fn add_command(&mut self, command: circuit_json::Command) -> Result<(), OpConvertError> {
        let circuit_json::Command {
            op, args, opgroup, ..
        } = command;
        let op_params = op.params.clone().unwrap_or_default();

        // Interpret the serialised operation as a [`Tk1Op`].
        let num_qubits = args
            .iter()
            .take_while(|&arg| self.is_qubit_register(arg))
            .count();
        let num_input_bits = args.len() - num_qubits;
        let tk1op = Tk1Op::from_serialised_op(op, num_qubits, num_input_bits);

        let (input_wires, output_registers) = self.get_op_wires(&tk1op, &args, op_params)?;
        let op: OpType = (&tk1op).into();

        let new_op = self.hugr.add_dataflow_op(op, input_wires).unwrap();
        let wires = new_op.outputs();

        // Store the opgroup metadata.
        if let Some(opgroup) = opgroup {
            self.hugr
                .set_child_metadata(new_op.node(), METADATA_OPGROUP, json!(opgroup));
        }

        // Assign the new output wires to some register, replacing the previous association.
        for (register, wire) in output_registers.into_iter().zip_eq(wires) {
            self.set_register_wire(register, wire);
        }

        Ok(())
    }

    /// Returns the input wires to connect to a new operation
    /// and the registers to associate with outputs.
    ///
    /// It may add constant nodes to the Hugr if the operation has constant parameters.
    fn get_op_wires(
        &mut self,
        tk1op: &Tk1Op,
        args: &[register::ElementId],
        params: Vec<String>,
    ) -> Result<(Vec<Wire>, Vec<RegisterHash>), OpConvertError> {
        // Arguments are always ordered with qubits first, and then bits.
        let mut inputs: Vec<Wire> = Vec::with_capacity(args.len() + params.len());
        let mut outputs: Vec<RegisterHash> =
            Vec::with_capacity(tk1op.qubit_outputs() + tk1op.bit_outputs());

        let mut current_arg = 0;
        let mut next_arg = || {
            if args.len() <= current_arg {
                return Err(OpConvertError::MissingSerialisedArguments {
                    optype: tk1op.optype(),
                    expected_qubits: tk1op.qubit_inputs(),
                    expected_bits: tk1op.bit_inputs(),
                    args: args.to_owned(),
                });
            }
            current_arg += 1;
            Ok(&args[current_arg - 1])
        };

        // Qubit wires
        assert_eq!(
            tk1op.qubit_inputs(),
            tk1op.qubit_outputs(),
            "Operations with different numbers of input and output qubits are not currently supported."
        );
        for _ in 0..tk1op.qubit_inputs() {
            let reg = next_arg()?;
            inputs.push(self.register_wire(reg));
            outputs.push(reg.into());
        }

        // Bit wires
        for zip in (0..tk1op.bit_inputs()).zip_longest(0..tk1op.bit_outputs()) {
            let reg = next_arg()?;
            match zip {
                EitherOrBoth::Both(_inp, _out) => {
                    // A bit used both as input and output.
                    inputs.push(self.register_wire(reg));
                    outputs.push(reg.into());
                }
                EitherOrBoth::Left(_inp) => {
                    // A bit used only used as input.
                    inputs.push(self.register_wire(reg));
                }
                EitherOrBoth::Right(_out) => {
                    // A new bit output.
                    outputs.push(reg.into());
                }
            }
        }

        // Check that the operation is not missing parameters.
        //
        // Nb: `Tk1Op::Opaque` operations may not have parameters in their hugr definition.
        // In that case, we just store the parameter values in the opaque data.
        if tk1op.num_params() > params.len() {
            return Err(OpConvertError::MissingSerialisedParams {
                optype: tk1op.optype(),
                expected: tk1op.num_params(),
                params,
            });
        }
        // Add the parameter wires to the input.
        inputs.extend(
            tk1op
                .param_ports()
                .zip(params)
                .map(|(_port, param)| self.load_parameter(param)),
        );

        Ok((inputs, outputs))
    }

    /// Returns the wire carrying a parameter.
    ///
    /// If the parameter is a constant, a constant definition is added to the Hugr.
    ///
    /// The returned wires always have float type.
    fn load_parameter(&mut self, param: String) -> Wire {
        fn process(
            hugr: &mut FunctionBuilder<Hugr>,
            input_params: &mut IndexMap<String, LoadedParameter>,
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
                        let wire = hugr.add_input(rotation_type());
                        LoadedParameter::rotation(wire)
                    })
                }
                PytketParam::Operation { op, args } => {
                    // We assume all operations take float inputs.
                    let input_wires = args
                        .into_iter()
                        .map(|arg| process(hugr, input_params, arg, param).as_float(hugr).wire)
                        .collect_vec();
                    let res = hugr.add_dataflow_op(op, input_wires).unwrap_or_else(|e| {
                        panic!("Error while decoding pytket operation parameter \"{param}\". {e}",)
                    });
                    assert_eq!(res.num_value_outputs(), 1, "An operation decoded from the pytket op parameter \"{param}\" had {} outputs", res.num_value_outputs());
                    LoadedParameter::float(res.out_wire(0))
                }
            }
        }

        let parsed = parse_pytket_param(&param);
        process(&mut self.hugr, &mut self.parameters, parsed, &param)
            .as_rotation(&mut self.hugr)
            .wire
    }

    /// Return the [`Wire`] associated with a register.
    fn register_wire(&self, register: impl Into<RegisterHash>) -> Wire {
        self.register_wires[&register.into()]
    }

    /// Update the tracked [`Wire`] for a register.
    fn set_register_wire(&mut self, register: impl Into<RegisterHash>, unit: Wire) {
        self.register_wires.insert(register.into(), unit);
    }

    /// Returns `true` if the register is a qubit register.
    fn is_qubit_register(&self, register: impl Into<RegisterHash>) -> bool {
        self.qubit_registers.contains(&register.into())
    }
}

/// Only single-indexed registers are supported.
fn check_register(register: &register::ElementId) -> Result<(), TK1ConvertError> {
    if register.1.len() != 1 {
        Err(TK1ConvertError::MultiIndexedRegister {
            register: register.0.clone(),
        })
    } else {
        Ok(())
    }
}

/// The type of a loaded parameter in the Hugr.
#[derive(Debug, Display, Clone, Copy, Hash, PartialEq, Eq)]
enum LoadedParameterType {
    Float,
    Rotation,
}

/// A loaded parameter in the Hugr.
///
/// Tracking the type of the wire lets us delay conversion between the types
/// until they are actually needed.
#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
struct LoadedParameter {
    /// The type of the parameter.
    pub typ: LoadedParameterType,
    /// The wire where the parameter is loaded.
    pub wire: Wire,
}

impl LoadedParameter {
    /// Returns a `LoadedParameter` for a float param.
    pub fn float(wire: Wire) -> LoadedParameter {
        LoadedParameter {
            typ: LoadedParameterType::Float,
            wire,
        }
    }

    /// Returns a `LoadedParameter` for a rotation param.
    pub fn rotation(wire: Wire) -> LoadedParameter {
        LoadedParameter {
            typ: LoadedParameterType::Rotation,
            wire,
        }
    }

    /// Convert the parameter into a given type, if necessary.
    ///
    /// Adds the necessary operations to the Hugr and returns a new wire.
    pub fn as_type(
        &self,
        typ: LoadedParameterType,
        hugr: &mut FunctionBuilder<Hugr>,
    ) -> LoadedParameter {
        match (self.typ, typ) {
            (LoadedParameterType::Float, LoadedParameterType::Rotation) => {
                let wire = hugr
                    .add_dataflow_op(RotationOp::from_halfturns_unchecked, [self.wire])
                    .unwrap()
                    .out_wire(0);
                LoadedParameter::rotation(wire)
            }
            (LoadedParameterType::Rotation, LoadedParameterType::Float) => {
                let wire = hugr
                    .add_dataflow_op(RotationOp::to_halfturns, [self.wire])
                    .unwrap()
                    .out_wire(0);
                LoadedParameter::float(wire)
            }
            _ => {
                debug_assert_eq!(self.typ, typ, "cannot convert {} to {}", self.typ, typ);
                *self
            }
        }
    }

    /// Convert the parameter into a float, if necessary.
    ///
    /// Adds the necessary operations to the Hugr and returns a new wire.
    pub fn as_float(&self, hugr: &mut FunctionBuilder<Hugr>) -> LoadedParameter {
        self.as_type(LoadedParameterType::Float, hugr)
    }

    /// Convert the parameter into a rotation, if necessary.
    ///
    /// Adds the necessary operations to the Hugr and returns a new wire.
    pub fn as_rotation(&self, hugr: &mut FunctionBuilder<Hugr>) -> LoadedParameter {
        self.as_type(LoadedParameterType::Rotation, hugr)
    }
}
