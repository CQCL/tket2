//! Intermediate structure for converting encoding [`Circuit`]s into [`SerialCircuit`]s.

use std::collections::HashMap;

use hugr::algorithm::const_fold::fold_const;
use hugr::extension::prelude::QB_T;
use hugr::ops::{Const, OpName, OpType};
use hugr::std_extensions::arithmetic::float_types::ConstF64;
use hugr::values::Value;
use hugr::Wire;
use itertools::{Either, Itertools};
use tket_json_rs::circuit_json::{self, Permutation, Register, SerialCircuit};

use crate::circuit::command::{CircuitUnit, Command};
use crate::circuit::Circuit;
use crate::extension::{LINEAR_BIT, SYM_OP_ID, TKET2_EXTENSION_ID};
use crate::ops::match_symb_const_op;

use super::op::JsonOp;
use super::{
    OpConvertError, METADATA_B_REGISTERS, METADATA_IMPLICIT_PERM, METADATA_PHASE,
    METADATA_Q_REGISTERS,
};

/// The state of an in-progress [`SerialCircuit`] being built from a [`Circuit`].
#[derive(Debug, PartialEq)]
pub(super) struct JsonEncoder {
    /// The name of the circuit being encoded.
    name: Option<String>,
    /// Global phase value. Defaults to "0"
    phase: String,
    /// Implicit permutation of output qubits
    implicit_permutation: Vec<Permutation>,
    /// The current encoded commands
    commands: Vec<circuit_json::Command>,
    /// The TKET1 qubit registers associated to each qubit unit of the circuit.
    qubit_to_reg: HashMap<CircuitUnit, Register>,
    /// The TKET1 bit registers associated to each linear bit unit of the circuit.
    bit_to_reg: HashMap<CircuitUnit, Register>,
    /// The ordered TKET1 names for the input qubit registers.
    qubit_registers: Vec<Register>,
    /// The ordered TKET1 names for the input bit registers.
    bit_registers: Vec<Register>,
    /// A register of constant values carried over non-linear wires. These may
    /// be variable names, or constant-folded values.
    constants: HashMap<Wire, EncodableConst>,
}

/// A constant value that can be encoded as a [`SerialCircuit`] parameter.
///
/// We keep constants as [`Const`] values until we need to serialize them,
/// so they can be constant-folded if possible.
///
/// If we can't constant-fold a value, we encode it as a symbolic string.
#[derive(Debug, Clone, PartialEq)]
enum EncodableConst {
    /// A Hugr constant value that can be constant-folded as needed.
    Const(Const),
    /// An already-encoded symbolic constant.
    Symbolic(String),
}

impl JsonEncoder {
    /// Create a new [`JsonEncoder`] from a [`Circuit`].
    pub fn new(circ: &impl Circuit) -> Self {
        let name = circ.name().map(str::to_string);

        let mut qubit_registers = vec![];
        let mut bit_registers = vec![];
        let mut phase = "0".to_string();
        let mut implicit_permutation = vec![];

        // Recover other parameters stored in the metadata
        // TODO: Check for invalid encoded metadata
        let root = circ.root();
        if let Some(p) = circ.get_metadata(root, METADATA_PHASE) {
            phase = p.as_str().unwrap().to_string();
        }
        if let Some(perm) = circ.get_metadata(root, METADATA_IMPLICIT_PERM) {
            implicit_permutation = serde_json::from_value(perm.clone()).unwrap();
        }
        if let Some(q_regs) = circ.get_metadata(root, METADATA_Q_REGISTERS) {
            qubit_registers = serde_json::from_value(q_regs.clone()).unwrap();
        }
        if let Some(b_regs) = circ.get_metadata(root, METADATA_B_REGISTERS) {
            bit_registers = serde_json::from_value(b_regs.clone()).unwrap();
        }

        // Map the Hugr units to tket1 register names.
        // Uses the names from the metadata if available, or initializes new sequentially-numbered registers.
        let mut bit_to_reg = HashMap::new();
        let mut qubit_to_reg = HashMap::new();
        let get_register = |registers: &mut Vec<Register>, prefix: &str, index| {
            registers.get(index).cloned().unwrap_or_else(|| {
                let r = Register(prefix.to_string(), vec![index as i64]);
                registers.push(r.clone());
                r
            })
        };
        for (unit, _, ty) in circ.units() {
            if ty == QB_T {
                let index = qubit_to_reg.len();
                let reg = get_register(&mut qubit_registers, "q", index);
                qubit_to_reg.insert(unit, reg);
            } else if ty == *LINEAR_BIT {
                let index = bit_to_reg.len();
                let reg = get_register(&mut bit_registers, "b", index);
                bit_to_reg.insert(unit, reg.clone());
            }
        }

        Self {
            name,
            phase,
            implicit_permutation,
            commands: vec![],
            qubit_to_reg,
            bit_to_reg,
            qubit_registers,
            bit_registers,
            constants: HashMap::new(),
        }
    }

    /// Add a circuit command to the serialization.
    pub fn add_command<C: Circuit>(
        &mut self,
        circ: &C,
        command: Command<'_, C>,
        optype: &OpType,
    ) -> Result<(), OpConvertError> {
        // Register the constant output of the command, if any.
        // This may require constant-folding the command (if possible).
        //
        // May error out if the constant output cannot be computed (e.g. if the
        // inputs are not all constants).
        if self.record_constants(circ, &command, optype) {
            // If the command was constant-folded, we don't add as a circuit gate.
            return Ok(());
        }

        let (args, input_params): (Vec<Register>, Vec<Wire>) =
            command
                .inputs()
                .partition_map(|(u, _, _)| match self.unit_to_register(u) {
                    Some(r) => Either::Left(r),
                    None => match u {
                        CircuitUnit::Wire(w) => Either::Right(w),
                        CircuitUnit::Linear(_) => {
                            panic!("No register found for the linear input {u:?} to command {command:?}.")
                        }
                    },
                });

        // TODO Restore the opgroup (once the decoding supports it)
        let opgroup = None;
        let op: JsonOp = optype.try_into()?;
        let mut op: circuit_json::Operation = op.into_operation();
        if !input_params.is_empty() {
            op.params = Some(
                input_params
                    .into_iter()
                    .filter_map(|w| self.constants.get(&w).cloned())
                    .map(|c| c.into_string())
                    .collect_vec(),
            )
        }
        // TODO: ops that contain free variables.
        // (update decoder to ignore them too, but store them in the wrapped op)

        let command = circuit_json::Command { op, args, opgroup };
        self.commands.push(command);
        Ok(())
    }

    pub fn finish(self) -> SerialCircuit {
        SerialCircuit {
            name: self.name,
            phase: self.phase,
            commands: self.commands,
            qubits: self.qubit_registers,
            bits: self.bit_registers,
            implicit_permutation: self.implicit_permutation,
        }
    }

    /// Record the constant outputs of the command.
    ///
    /// If the output can be computed from the constant inputs, the command is
    /// evaluated using the commands's [`ConstFold`] method.
    ///
    /// Returns `true` if any parameter was recorded.
    fn record_constants<C: Circuit>(
        &mut self,
        circ: &C,
        command: &Command<'_, C>,
        optype: &OpType,
    ) -> bool {
        // Only consider commands where all inputs are parameters.
        let input_ports = circ.node_inputs(command.node());
        let inputs = command
            .inputs()
            .zip(input_ports)
            .filter_map(|((unit, _, _), inp_port)| match unit {
                CircuitUnit::Wire(wire) => Some((inp_port, self.constants.get(&wire)?)),
                CircuitUnit::Linear(_) => None,
            })
            .collect_vec();

        // If the command has non-constant inputs, we can't fold it.
        // (It will be added as a circuit gate instead).
        if inputs.len() != command.input_count() {
            debug_assert!(!matches!(
                optype,
                OpType::Const(_) | OpType::LoadConstant(_)
            ));
            return false;
        }

        // Special cases for operations that define a new constant.
        match optype {
            // Hugr constant definition.
            OpType::Const(constant) => {
                // New constant definition
                for (_, wire) in command.output_wires() {
                    self.add_constant(wire, constant.clone());
                }
                return true;
            }
            // tket1 symbolic constant definition (already encoded).
            OpType::LeafOp(op) if op.name() == format!("{TKET2_EXTENSION_ID}.{SYM_OP_ID}") => {
                let symbol = match_symb_const_op(optype).unwrap_or_else(|| {
                    panic!("Unable to extract the symbolic constant from optype {optype:#?}.")
                });
                for (_, wire) in command.output_wires() {
                    self.constants
                        .insert(wire, EncodableConst::Symbolic(symbol.clone()));
                }
                return true;
            }
            // A lifting from a constant wire into a value. We can propagate the constant directly.
            OpType::LoadConstant(_) => {
                debug_assert_eq!(inputs.len(), 1);
                let constant = inputs[0].1.clone();
                for (_, wire) in command.output_wires() {
                    self.constants.insert(wire, constant.clone());
                }
                return true;
            }
            _ => {}
        }

        // Fold the command, if possible.
        // Encode them as symbolic constants otherwise.
        if inputs
            .iter()
            .all(|(_, c)| matches!(c, EncodableConst::Const(_)))
        {
            // All inputs are constants, so we can constant-fold the command.
            // (This may still fail if the command is not constant-foldable).
            let inputs = inputs
                .into_iter()
                .map(|(p, c)| match c {
                    EncodableConst::Const(c) => (p, c.clone()),
                    EncodableConst::Symbolic(_) => unreachable!(),
                })
                .collect_vec();
            let consts = fold_const(optype, &inputs).unwrap_or_else(|| {
                // This command cannot be constant-folded.
                panic!("Unable to constant-fold command {command:#?} with optype {optype:#?}.")
            });
            for (out_port, constant) in consts {
                let wire = Wire::new(command.node(), out_port);
                self.add_constant(wire, constant);
            }
        } else {
            // Some inputs are symbolic constants, so we can't constant-fold.
            //
            // TODO: Add support for symbolic folding in HUGR.
            panic!("Unable to constant-fold command {command:#?} with optype {optype:#?}.")
        }
        true
    }

    /// Associate a constant value with a non-linear wire.
    pub(super) fn add_constant(&mut self, wire: Wire, constant: Const) {
        self.constants.insert(wire, EncodableConst::Const(constant));
    }

    /// Translate a linear [`CircuitUnit`] into a [`Register`], if possible.
    fn unit_to_register(&self, unit: CircuitUnit) -> Option<Register> {
        self.qubit_to_reg
            .get(&unit)
            .or_else(|| self.bit_to_reg.get(&unit))
            .cloned()
    }
}

impl EncodableConst {
    /// Encode a constant value as a string that can be added to a [`SerialCircuit`].
    #[allow(unused)]
    fn into_encoded(self) -> Self {
        Self::Symbolic(self.into_string())
    }

    /// The string-encoded value of the constant.
    fn into_string(self) -> String {
        let constant = match self {
            Self::Symbolic(s) => return s,
            Self::Const(c) => c,
        };

        // Encode the constant as a JSON string.
        //
        // If the constant is an extension value, we encode the internal data directly.
        //
        // TODO: Ideally, the CustomConst would be able to encode themselves as a string.
        // Here we have to manually hardcode the float conversion instead.
        match constant.value() {
            Value::Extension { c: (val,) } => {
                if let Some(f) = val.downcast_ref::<ConstF64>() {
                    f.to_string()
                } else {
                    serde_json::to_string(val).unwrap()
                }
            }
            _ => serde_json::to_string(&constant).unwrap(),
        }
    }
}
