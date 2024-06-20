//! Intermediate structure for converting decoding [`SerialCircuit`]s into [`Hugr`]s.

use std::collections::hash_map::DefaultHasher;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::mem;

use hugr::builder::{CircuitBuilder, Container, Dataflow, DataflowHugr, FunctionBuilder};
use hugr::extension::prelude::{BOOL_T, QB_T};

use hugr::ops::OpType;
use hugr::types::FunctionType;
use hugr::CircuitUnit;
use hugr::{Hugr, Wire};

use itertools::{EitherOrBoth, Itertools};
use serde_json::json;
use tket_json_rs::circuit_json;
use tket_json_rs::circuit_json::SerialCircuit;

use super::op::Tk1Op;
use super::{
    try_param_to_constant, OpConvertError, TK1ConvertError, METADATA_IMPLICIT_PERM,
    METADATA_OPGROUP, METADATA_PHASE,
};
use super::{METADATA_B_REGISTERS, METADATA_Q_REGISTERS};
use crate::extension::{REGISTRY, TKET1_EXTENSION_ID};
use crate::symbolic_constant_op;

/// The state of an in-progress [`FunctionBuilder`] being built from a [`SerialCircuit`].
///
/// Mostly used to define helper internal methods.
#[derive(Debug, PartialEq)]
pub(super) struct Tk1Decoder {
    /// The Hugr being built.
    pub hugr: FunctionBuilder<Hugr>,
    /// The dangling wires of the builder.
    /// Used to generate [`CircuitBuilder`]s.
    dangling_wires: Vec<Wire>,
    /// A map from the json registers to the units in the circuit being built.
    register_units: HashMap<RegisterHash, CircuitUnit>,
}

impl Tk1Decoder {
    /// Initialize a new [`Tk1Decoder`], using the metadata from a [`SerialCircuit`].
    pub fn try_new(serialcirc: &SerialCircuit) -> Result<Self, TK1ConvertError> {
        let num_qubits = serialcirc.qubits.len();
        let num_bits = serialcirc.bits.len();
        let sig = FunctionType::new_endo([vec![QB_T; num_qubits], vec![BOOL_T; num_bits]].concat())
            .with_extension_delta(TKET1_EXTENSION_ID);

        let name = serialcirc.name.clone().unwrap_or_default();
        let mut dfg = FunctionBuilder::new(name, sig.into()).unwrap();
        let dangling_wires = dfg.input_wires().collect::<Vec<_>>();

        // Metadata. The circuit requires "name", and we store other things that
        // should pass through the serialization roundtrip.
        dfg.set_metadata(METADATA_PHASE, json!(serialcirc.phase));
        dfg.set_metadata(
            METADATA_IMPLICIT_PERM,
            json!(serialcirc.implicit_permutation),
        );
        dfg.set_metadata(METADATA_Q_REGISTERS, json!(serialcirc.qubits));
        dfg.set_metadata(METADATA_B_REGISTERS, json!(serialcirc.bits));

        // Map each register element to their starting `CircuitUnit`.
        let mut wire_map: HashMap<RegisterHash, CircuitUnit> =
            HashMap::with_capacity(num_bits + num_qubits);
        for (i, register) in serialcirc.qubits.iter().enumerate() {
            check_register(register)?;
            wire_map.insert(register.into(), CircuitUnit::Linear(i));
        }
        for (register, &input_wire) in serialcirc
            .bits
            .iter()
            .zip(dangling_wires.iter().skip(num_qubits))
        {
            check_register(register)?;
            wire_map.insert(register.into(), CircuitUnit::Wire(input_wire));
        }

        Ok(Tk1Decoder {
            hugr: dfg,
            dangling_wires,
            register_units: wire_map,
        })
    }

    /// Finish building the [`Hugr`].
    pub fn finish(self) -> Hugr {
        // TODO: Throw validation error?
        self.hugr
            .finish_hugr_with_outputs(self.dangling_wires, &REGISTRY)
            .unwrap()
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
            .take_while(|&arg| self.register_unit(arg).is_linear())
            .count();
        let num_input_bits = args.len() - num_qubits;
        let tk1op = Tk1Op::from_serialised_op(op, num_qubits, num_input_bits);

        let (input_wires, output_registers) = self.get_op_units(&tk1op, &args, op_params)?;
        let op: OpType = (&tk1op).into();

        let wires =
            self.with_circ_builder(|circ| circ.append_with_outputs(op, input_wires).unwrap());

        // Store the opgroup metadata.
        if let (Some(opgroup), [w, ..]) = (opgroup, wires.as_slice()) {
            self.hugr
                .set_child_metadata(w.node(), METADATA_OPGROUP, json!(opgroup));
        }

        // Assign the new output wires to some register, if needed.
        for (register, wire) in output_registers.into_iter().zip(wires) {
            self.set_register_unit(&register, wire);
        }

        Ok(())
    }

    /// Apply a function to the internal hugr builder viewed as a [`CircuitBuilder`].
    fn with_circ_builder<F, T>(&mut self, f: F) -> T
    where
        F: FnOnce(&mut CircuitBuilder<FunctionBuilder<Hugr>>) -> T,
    {
        let mut circ = self.hugr.as_circuit(mem::take(&mut self.dangling_wires));
        let res = f(&mut circ);
        self.dangling_wires = circ.finish();
        res
    }

    /// Returns the input units to connect to a new operation
    /// and optionally a register to associate with the new output wires.
    ///
    /// It may add constant nodes to the Hugr if the operation has constant parameters.
    fn get_op_units(
        &mut self,
        tk1op: &Tk1Op,
        args: &[circuit_json::Register],
        params: Vec<String>,
    ) -> Result<(Vec<CircuitUnit>, Vec<circuit_json::Register>), OpConvertError> {
        // Arguments are always ordered with qubits first, and then bits.
        let mut inputs = Vec::with_capacity(args.len() + params.len());
        let mut outputs = Vec::with_capacity(tk1op.qubit_outputs() + tk1op.bit_outputs());

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
            inputs.push(self.register_unit(&reg));
            outputs.push(reg.clone());
        }

        // Bit wires
        for zip in (0..tk1op.bit_inputs()).zip_longest(0..tk1op.bit_outputs()) {
            let reg = next_arg()?;
            match zip {
                EitherOrBoth::Both(_inp, _out) => {
                    // A bit used both as input and output.
                    inputs.push(self.register_unit(&reg));
                    outputs.push(reg.clone());
                }
                EitherOrBoth::Left(_inp) => {
                    // A bit used only used as input.
                    inputs.push(self.register_unit(&reg));
                }
                EitherOrBoth::Right(_out) => {
                    // A new bit output.
                    outputs.push(reg.clone());
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
        // Add the parameter wires to the units.
        inputs.extend(
            tk1op
                .param_ports()
                .zip(params.into_iter())
                .map(|(_port, param)| CircuitUnit::Wire(self.create_param_wire(&param))),
        );

        Ok((inputs, outputs))
    }

    /// Returns the wire carrying a parameter.
    ///
    /// If the parameter is a constant, a constant definition is added to the Hugr.
    ///
    /// TODO: If the parameter is a variable, returns the corresponding wire from the input.
    fn create_param_wire(&mut self, param: &str) -> Wire {
        match try_param_to_constant(param) {
            Some(const_op) => self.hugr.add_load_const(const_op),
            None => {
                // store string in custom op.
                let symb_op = symbolic_constant_op(param);
                let o = self.hugr.add_dataflow_op(symb_op, []).unwrap();
                o.out_wire(0)
            }
        }
    }

    /// Return the wire unit for the `elem`th value of a given register.
    ///
    /// Relies on TKET1 constraint that all registers have unique names.
    fn register_unit(&self, register: &circuit_json::Register) -> CircuitUnit {
        self.register_units[&register.into()]
    }

    /// Update the [`CircuitUnit`] for a register.
    fn set_register_unit(
        &mut self,
        register: &circuit_json::Register,
        unit: impl Into<CircuitUnit>,
    ) {
        self.register_units.insert(register.into(), unit.into());
    }
}

/// A hashed register, used to identify registers in the [`Tk1Decoder::register_wire`] map,
/// avoiding string clones on lookup.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct RegisterHash {
    hash: u64,
}

impl From<&circuit_json::Register> for RegisterHash {
    fn from(reg: &circuit_json::Register) -> Self {
        let mut hasher = DefaultHasher::new();
        reg.0.hash(&mut hasher);
        reg.1.hash(&mut hasher);
        Self {
            hash: hasher.finish(),
        }
    }
}

/// Only single-indexed registers are supported.
fn check_register(register: &circuit_json::Register) -> Result<(), TK1ConvertError> {
    if register.1.len() != 1 {
        Err(TK1ConvertError::MultiIndexedRegister {
            register: register.0.clone(),
        })
    } else {
        Ok(())
    }
}
