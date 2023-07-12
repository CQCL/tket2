//! Json serialization and deserialization.

pub mod op;

#[cfg(test)]
mod tests;

use std::collections::hash_map::DefaultHasher;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::mem;

use hugr::builder::{AppendWire, CircuitBuilder, Container, DFGBuilder, Dataflow, DataflowHugr};
use hugr::ops::{ConstValue, OpType};
use hugr::types::Signature;
use hugr::{Hugr, Wire};

use serde_json::json;
use thiserror::Error;
use tket_json_rs::circuit_json;
use tket_json_rs::circuit_json::SerialCircuit;
use tket_json_rs::optype::OpType as JsonOpType;

use self::op::JsonOp;
use crate::utils::{BIT, QB};

/// A JSON-serialized TKET1 circuit that can be converted to a [`Hugr`].
pub trait TKET1Decode {
    /// The error type for decoding.
    type DecodeError;
    /// Convert the serialized circuit to a [`Hugr`].
    fn decode(self) -> Result<Hugr, Self::DecodeError>;
}

impl TKET1Decode for SerialCircuit {
    type DecodeError = OpConvertError;
    fn decode(self) -> Result<Hugr, Self::DecodeError> {
        let mut decoder = JsonDecoder::new(&self);

        if !self.phase.is_empty() {
            // TODO - add a phase gate
            // let phase = Param::new(serialcirc.phase);
            // decoder.add_phase(phase);
        }

        // TODO: Check the implicit permutation in the serialized circuit.

        for com in self.commands {
            decoder.add_command(com);
        }
        Ok(decoder.finish())
    }
}

/// The state of an in-progress [`DFGBuilder`] being built from a [`SerialCircuit`].
///
/// Mostly used to define helper internal methods.
#[derive(Debug, PartialEq)]
struct JsonDecoder {
    /// The Hugr being built.
    pub hugr: DFGBuilder<Hugr>,
    /// The dangling wires of the builder.
    /// Used to generate [`CircuitBuilder`]s.
    dangling_wires: Vec<Wire>,
    /// A map from the json registers to flat wire indices.
    register_wire: HashMap<RegisterHash, usize>,
    /// The number of qubits in the circuit.
    num_qubits: usize,
    /// The number of bits in the circuit.
    num_bits: usize,
}

impl JsonDecoder {
    /// Initialize a new [`JsonDecoder`], using the metadata from a [`SerialCircuit`].
    pub fn new(serialcirc: &SerialCircuit) -> Self {
        let num_qubits = serialcirc.qubits.len();
        let num_bits = serialcirc.bits.len();
        // Map each (register name, index) pair to an offset in the signature.
        let mut wire_map: HashMap<RegisterHash, usize> =
            HashMap::with_capacity(num_bits + num_qubits);
        for (i, register) in serialcirc
            .qubits
            .iter()
            .chain(serialcirc.bits.iter())
            .enumerate()
        {
            if register.1.len() != 1 {
                // TODO: Support multi-index registers?
                panic!("Register {} has more than one index", register.0);
            }
            wire_map.insert((register, 0).into(), i);
        }
        let sig = Signature::new_linear([vec![QB; num_qubits], vec![BIT; num_bits]].concat());

        let mut dfg = DFGBuilder::new(sig.input, sig.output).unwrap();

        dfg.set_metadata(json!({"name": serialcirc.name}));

        let dangling_wires = dfg.input_wires().collect::<Vec<_>>();
        JsonDecoder {
            hugr: dfg,
            dangling_wires,
            register_wire: wire_map,
            num_qubits,
            num_bits,
        }
    }

    /// Finish building the [`Hugr`].
    pub fn finish(self) -> Hugr {
        // TODO: Throw validation error?
        self.hugr
            .finish_hugr_with_outputs(self.dangling_wires)
            .unwrap()
    }

    /// Add a [`Command`] from the serial circuit to the [`JsonDecoder`].
    ///
    /// - [`Command`]: circuit_json::Command
    pub fn add_command(&mut self, command: circuit_json::Command) {
        let circuit_json::Command { op, args, .. } = command;
        let params = op.params.clone().unwrap_or_default();
        let num_qubits = args
            .iter()
            .take_while(|&arg| self.reg_wire(arg, 0) < self.num_qubits)
            .count();
        let num_bits = args.len() - num_qubits;
        let op = JsonOp::new_from_op(op, num_qubits, num_bits);

        let args: Vec<_> = args.into_iter().map(|reg| self.reg_wire(&reg, 0)).collect();

        let param_wires: Vec<Wire> = params.iter().map(|p| self.get_param_wire(p)).collect();

        let append_wires = args
            .into_iter()
            .map(AppendWire::I)
            .chain(param_wires.into_iter().map(AppendWire::W));

        self.with_circ_builder(|circ| {
            circ.append_and_consume(&op, append_wires).unwrap();
        });
    }

    /// Apply a function to the internal hugr builder viewed as a [`CircuitBuilder`].
    fn with_circ_builder(&mut self, f: impl FnOnce(&mut CircuitBuilder<DFGBuilder<Hugr>>)) {
        let mut circ = self.hugr.as_circuit(mem::take(&mut self.dangling_wires));
        f(&mut circ);
        self.dangling_wires = circ.finish();
    }

    /// Returns the wire carrying a parameter.
    ///
    /// If the parameter is a constant, a constant definition is added to the Hugr.
    ///
    /// TODO: If the parameter is a variable, returns the corresponding wire from the input.
    fn get_param_wire(&mut self, param: &str) -> Wire {
        if let Ok(f) = param.parse::<f64>() {
            self.hugr.add_load_const(ConstValue::F64(f)).unwrap()
        } else if param.split('/').count() == 2 {
            // TODO: Use the rational types from `Hugr::extensions::rotation`
            let (n, d) = param.split_once('/').unwrap();
            let n = n.parse::<f64>().unwrap();
            let d = d.parse::<f64>().unwrap();
            self.hugr.add_load_const(ConstValue::F64(n / d)).unwrap()
        } else {
            // TODO: Pre-compute variables and add them to the input signature.
            todo!("Variable parameters not yet supported")
        }
    }

    /// Return the wire index for the `elem`th value of a given register.
    ///
    /// Relies on TKET1 constraint that all registers have unique names.
    fn reg_wire(&self, register: &circuit_json::Register, elem: usize) -> usize {
        self.register_wire[&(register, elem).into()]
    }
}

/// The state of an in-progress [`SerialCircuit`] being built from a [`Circuit`].
#[derive(Debug, PartialEq)]
struct JsonEncoder {
    /// The Hugr being built.
    pub serial: SerialCircuit,
    /// The dangling wires of the builder.
    /// Used to generate [`CircuitBuilder`]s.
    dangling_wires: Vec<Wire>,
    /// A map from the json registers to flat wire indices.
    register_wire: HashMap<RegisterHash, usize>,
    /// The number of qubits in the circuit.
    num_qubits: usize,
    /// The number of bits in the circuit.
    num_bits: usize,
}

//impl JsonEncoder {
//    /// Create a new [`JsonEncoder`] from a [`Circuit`].
//    pub fn new(circ: &impl Circuit) -> Self {
//        let num_qubits = circ.qubits().len();
//        let num_bits = circ.bits().len();
//    }
//}

/// Error type for conversion between `Op` and `OpType`.
#[derive(Debug, Error)]
pub enum OpConvertError {
    /// The serialized operation is not supported.
    #[error("Unsupported serialized operation: {0:?}")]
    UnsupportedSerializedOp(JsonOpType),
    /// The serialized operation is not supported.
    #[error("Cannot serialize operation: {0:?}")]
    UnsupportedOpSerialization(OpType),
}

/// A hashed register, used to identify registers in the [`JsonDecoder::register_wire`] map,
/// avoiding string clones on lookup.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct RegisterHash {
    hash: u64,
}

impl From<(&circuit_json::Register, usize)> for RegisterHash {
    fn from((reg, elem): (&circuit_json::Register, usize)) -> Self {
        let mut hasher = DefaultHasher::new();
        reg.0.hash(&mut hasher);
        reg.1[elem].hash(&mut hasher);
        Self {
            hash: hasher.finish(),
        }
    }
}
