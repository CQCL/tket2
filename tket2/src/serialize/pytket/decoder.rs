//! Intermediate structure for converting decoding [`SerialCircuit`]s into [`Hugr`]s.

use std::collections::hash_map::DefaultHasher;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::mem;

use hugr::builder::{CircuitBuilder, Container, Dataflow, DataflowHugr, FunctionBuilder};
use hugr::extension::prelude::QB_T;

use hugr::ops::OpType;
use hugr::types::FunctionType;
use hugr::CircuitUnit;
use hugr::{Hugr, Wire};

use itertools::Itertools;
use serde_json::json;
use tket_json_rs::circuit_json;
use tket_json_rs::circuit_json::SerialCircuit;

use super::op::Tk1Op;
use super::{try_param_to_constant, TK1ConvertError, METADATA_IMPLICIT_PERM, METADATA_PHASE};
use super::{METADATA_B_REGISTERS, METADATA_Q_REGISTERS};
use crate::extension::{LINEAR_BIT, REGISTRY, TKET1_EXTENSION_ID};
use crate::symbolic_constant_op;

/// The state of an in-progress [`FunctionBuilder`] being built from a [`SerialCircuit`].
///
/// Mostly used to define helper internal methods.
#[derive(Debug, PartialEq)]
pub(super) struct JsonDecoder {
    /// The Hugr being built.
    pub hugr: FunctionBuilder<Hugr>,
    /// The dangling wires of the builder.
    /// Used to generate [`CircuitBuilder`]s.
    dangling_wires: Vec<Wire>,
    /// A map from the json registers to the units in the circuit being built.
    register_units: HashMap<RegisterHash, CircuitUnit>,
    /// The number of qubits in the circuit.
    num_qubits: usize,
    /// The number of bits in the circuit.
    num_bits: usize,
}

impl JsonDecoder {
    /// Initialize a new [`JsonDecoder`], using the metadata from a [`SerialCircuit`].
    pub fn try_new(serialcirc: &SerialCircuit) -> Result<Self, TK1ConvertError> {
        let num_qubits = serialcirc.qubits.len();
        let num_bits = serialcirc.bits.len();
        let sig = FunctionType::new_endo(
            [vec![QB_T; num_qubits], vec![LINEAR_BIT.clone(); num_bits]].concat(),
        )
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
        for (i, register) in serialcirc.bits.iter().enumerate() {
            check_register(register)?;
            wire_map.insert(register.into(), CircuitUnit::Linear(i + num_qubits));
        }

        Ok(JsonDecoder {
            hugr: dfg,
            dangling_wires,
            register_units: wire_map,
            num_qubits,
            num_bits,
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
    pub fn add_command(&mut self, command: circuit_json::Command) {
        // TODO Store the command's `opgroup` in the metadata.
        let circuit_json::Command { op, args, .. } = command;
        let num_qubits = args
            .iter()
            .take_while(|&arg| match self.reg_wire(arg) {
                CircuitUnit::Linear(i) => i < self.num_qubits,
                _ => false,
            })
            .count();
        let num_input_bits = args.len() - num_qubits;
        let op_params = op.params.clone();
        let tk1op = Tk1Op::from_serialised_op(op, num_qubits, num_input_bits);

        let param_units = tk1op
            .param_ports()
            .enumerate()
            .filter_map(|(i, _port)| op_params.as_ref()?.get(i).map(String::as_ref))
            .map(|p| CircuitUnit::Wire(self.create_param_wire(p)))
            .collect_vec();
        let arg_units = args.into_iter().map(|reg| self.reg_wire(&reg));

        let append_wires: Vec<CircuitUnit> = arg_units.chain(param_units).collect_vec();
        let op: OpType = (&tk1op).into();

        self.with_circ_builder(|circ| {
            circ.append_and_consume(op, append_wires).unwrap();
        });
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
    fn reg_wire(&self, register: &circuit_json::Register) -> CircuitUnit {
        self.register_units[&register.into()]
    }
}

/// A hashed register, used to identify registers in the [`JsonDecoder::register_wire`] map,
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
