//! Operations that have corresponding representations in both `pytket` and `tket2`.

use hugr::extension::prelude::QB_T;

use hugr::ops::{Noop, OpTrait, OpType};
use hugr::std_extensions::arithmetic::float_types::FLOAT64_TYPE;
use hugr::types::FunctionType;

use hugr::IncomingPort;
use tket_json_rs::circuit_json;
use tket_json_rs::optype::OpType as Tk1OpType;

use crate::extension::LINEAR_BIT;
use crate::Tk2Op;

/// An operation with a native TKET2 counterpart.
///
/// Note that the signature of the native and serialised operations may differ.
#[derive(Clone, Debug, PartialEq)]
pub struct NativeOp {
    /// The tket2 optype.
    op: OpType,
    /// The corresponding serialised optype.
    ///
    /// Some specific operations do not have a direct pytket counterpart, and must be handled
    /// separately.
    serial_op: Option<Tk1OpType>,
}

impl NativeOp {
    /// Create a new `NativeOp` from a `circuit_json::Operation`.
    pub fn try_from_tk2op(tk2op: Tk2Op) -> Option<Self> {
        let serial_op = match tk2op {
            Tk2Op::H => Tk1OpType::H,
            Tk2Op::CX => Tk1OpType::CX,
            Tk2Op::T => Tk1OpType::T,
            Tk2Op::S => Tk1OpType::S,
            Tk2Op::X => Tk1OpType::X,
            Tk2Op::Y => Tk1OpType::Y,
            Tk2Op::Z => Tk1OpType::Z,
            Tk2Op::Tdg => Tk1OpType::Tdg,
            Tk2Op::Sdg => Tk1OpType::Sdg,
            Tk2Op::ZZMax => Tk1OpType::ZZMax,
            Tk2Op::RzF64 => Tk1OpType::Rz,
            Tk2Op::RxF64 => Tk1OpType::Rx,
            Tk2Op::TK1 => Tk1OpType::TK1,
            Tk2Op::PhasedX => Tk1OpType::PhasedX,
            Tk2Op::ZZPhase => Tk1OpType::ZZPhase,
            Tk2Op::CZ => Tk1OpType::CZ,
            Tk2Op::Reset => Tk1OpType::Reset,
            Tk2Op::AngleAdd => {
                // These operations should be folded into constant before serialisation,
                // or replaced by pytket logic expressions.
                return Some(Self {
                    op: tk2op.into(),
                    serial_op: None,
                });
            }
            // TKET2 measurements and TKET1 measurements have different semantics.
            Tk2Op::Measure => {
                return None;
            }
            // These operations do not have a direct pytket counterpart.
            Tk2Op::QAlloc | Tk2Op::QFree => {
                return None;
            }
        };

        Some(Self {
            op: tk2op.into(),
            serial_op: Some(serial_op),
        })
    }

    /// Returns the translated tket2 optype for this operation, if it exists.
    pub fn try_from_serial_optype(serial_op: Tk1OpType) -> Option<Self> {
        let op = match serial_op {
            Tk1OpType::H => Tk2Op::H.into(),
            Tk1OpType::CX => Tk2Op::CX.into(),
            Tk1OpType::T => Tk2Op::T.into(),
            Tk1OpType::Tdg => Tk2Op::Tdg.into(),
            Tk1OpType::X => Tk2Op::X.into(),
            Tk1OpType::Y => Tk2Op::Y.into(),
            Tk1OpType::Z => Tk2Op::Z.into(),
            Tk1OpType::Rz => Tk2Op::RzF64.into(),
            Tk1OpType::Rx => Tk2Op::RxF64.into(),
            Tk1OpType::TK1 => Tk2Op::TK1.into(),
            Tk1OpType::PhasedX => Tk2Op::PhasedX.into(),
            Tk1OpType::ZZMax => Tk2Op::ZZMax.into(),
            Tk1OpType::ZZPhase => Tk2Op::ZZPhase.into(),
            Tk1OpType::CZ => Tk2Op::CZ.into(),
            Tk1OpType::Reset => Tk2Op::Reset.into(),
            Tk1OpType::noop => Noop::new(QB_T).into(),
            _ => {
                return None;
            }
        };
        Some(Self {
            op,
            serial_op: Some(serial_op),
        })
    }

    /// Converts this `NativeOp` into a tket_json_rs operation.
    pub fn serialised_op(&self) -> Option<circuit_json::Operation> {
        let serial_op = self.serial_op.clone()?;

        let mut num_qubits = 0;
        let mut num_bits = 0;
        let mut num_params = 0;
        if let Some(sig) = self.signature() {
            for ty in sig.input.iter() {
                if ty == &QB_T {
                    num_qubits += 1
                } else if *ty == *LINEAR_BIT {
                    num_bits += 1
                } else if ty == &FLOAT64_TYPE {
                    num_params += 1
                }
            }
        }

        let params = (num_params > 0).then(|| vec!["".into(); num_params]);

        Some(circuit_json::Operation {
            op_type: serial_op,
            n_qb: Some(num_qubits as u32),
            params,
            op_box: None,
            signature: Some([vec!["Q".into(); num_qubits], vec!["B".into(); num_bits]].concat()),
            conditional: None,
        })
    }

    /// Returns the dataflow signature for this operation.
    pub fn signature(&self) -> Option<FunctionType> {
        self.op.dataflow_signature()
    }

    /// Returns the tket2 optype for this operation.
    pub fn optype(&self) -> &OpType {
        &self.op
    }

    /// Consumes the `NativeOp` and returns the underlying `OpType`.
    pub fn into_op(self) -> OpType {
        self.op
    }

    /// Returns the ports corresponding to parameters for this operation.
    pub fn param_ports(&self) -> impl Iterator<Item = IncomingPort> + '_ {
        self.signature().into_iter().flat_map(|sig| {
            let types = sig.input_types().to_owned();
            sig.input_ports()
                .zip(types)
                .filter(|(_, ty)| ty == &FLOAT64_TYPE)
                .map(|(port, _)| port)
        })
    }
}
