//! Encoder and decoder for tket2 operations with native pytket counterparts.

use std::borrow::Cow;

use hugr::extension::prelude::{bool_t, qb_t, Noop};

use hugr::extension::ExtensionId;
use hugr::ops::{OpTrait, OpType};
use hugr::std_extensions::arithmetic::float_types::float64_type;
use hugr::types::Signature;

use hugr::{HugrView, IncomingPort, Wire};
use tket_json_rs::optype::OpType as Tk1OpType;

use crate::extension::rotation::rotation_type;
use crate::extension::TKET2_EXTENSION;
use crate::serialize::pytket::encoder::{Tk1Encoder, Tk1EncoderContext};
use crate::serialize::pytket::Tk1ConvertError;
use crate::{Circuit, Tk2Op};

/// Encoder for [Tk2Op] operations.
#[derive(Debug, Clone, Default)]
pub struct Tk2OpEncoder;

impl<H: HugrView> Tk1Encoder<H> for Tk2OpEncoder {
    fn extensions(&self) -> Vec<Cow<'_, ExtensionId>> {
        vec![Cow::Borrowed(TKET2_EXTENSION.name())]
    }

    fn op_to_pytket(
        &self,
        node: H::Node,
        op: &OpType,
        circ: &Circuit<H>,
        encoder: &mut Tk1EncoderContext<H>,
    ) -> Result<bool, Tk1ConvertError<H::Node>> {
        let Some(tk2op): Option<Tk2Op> = op.cast() else {
            return Ok(false);
        };

        let serial_op = match tk2op {
            Tk2Op::H => Tk1OpType::H,
            Tk2Op::CX => Tk1OpType::CX,
            Tk2Op::CY => Tk1OpType::CY,
            Tk2Op::CZ => Tk1OpType::CZ,
            Tk2Op::CRz => Tk1OpType::CRz,
            Tk2Op::T => Tk1OpType::T,
            Tk2Op::Tdg => Tk1OpType::Tdg,
            Tk2Op::S => Tk1OpType::S,
            Tk2Op::Sdg => Tk1OpType::Sdg,
            Tk2Op::X => Tk1OpType::X,
            Tk2Op::Y => Tk1OpType::Y,
            Tk2Op::Z => Tk1OpType::Z,
            Tk2Op::Rx => Tk1OpType::Rx,
            Tk2Op::Rz => Tk1OpType::Rz,
            Tk2Op::Ry => Tk1OpType::Ry,
            Tk2Op::Toffoli => Tk1OpType::CCX,
            Tk2Op::Reset => Tk1OpType::Reset,
            Tk2Op::Measure => Tk1OpType::Measure,
            // We translate `MeasureFree` the same way as a `Measure` operation.
            // Since the node does not have outputs the qubit/bit will simply be ignored,
            // but will appear when collecting the final pytket registers.
            Tk2Op::MeasureFree => Tk1OpType::Measure,
            // These operations are implicitly supported by the encoding,
            // they do not create a new command but just modify the value trackers.
            Tk2Op::QAlloc => {
                let out_port = circ.hugr().node_outputs(node).next().unwrap();
                let wire = Wire::new(node, out_port);
                let qb = encoder.values.new_qubit();
                encoder.values.register_values(wire, [qb], circ)?;
                return Ok(true);
            }
            // Since the qubit still gets connected at the end of the circuit,
            // `QFree` is a no-op.
            Tk2Op::QFree => {
                return Ok(true);
            }
            // Unsupported
            Tk2Op::TryQAlloc => {
                return Ok(false);
            }
        };

        // Most operations map directly to a pytket one.
        encoder.emit_command_for_node(serial_op, node, circ)?;

        Ok(true)
    }
}

/// An operation with a native TKET2 counterpart.
///
/// Note that the signature of the native and serialised operations may differ.
#[derive(Clone, Debug, PartialEq, Default)]
pub struct NativeOp {
    /// The tket2 optype.
    op: OpType,
    /// The corresponding serialised optype.
    ///
    /// Some specific operations do not have a direct pytket counterpart, and must be handled
    /// separately.
    serial_op: Option<Tk1OpType>,
    /// Number of input qubits to the operation.
    pub input_qubits: usize,
    /// Number of output qubits to the operation.
    pub input_bits: usize,
    /// Number of parameters to the operation.
    pub num_params: usize,
    /// Number of output qubits to the operation.
    pub output_qubits: usize,
    /// Number of output bits to the operation.
    pub output_bits: usize,
}

impl NativeOp {
    /// Initialise a new `NativeOp`.
    fn new(op: OpType, serial_op: Option<Tk1OpType>) -> Self {
        let mut native_op = Self {
            op,
            serial_op,
            ..Default::default()
        };
        native_op.compute_counts();
        native_op
    }

    /// Returns the translated tket2 optype for this operation, if it exists.
    pub fn try_from_serial_optype(serial_op: Tk1OpType) -> Option<Self> {
        let op = match serial_op {
            Tk1OpType::H => Tk2Op::H.into(),
            Tk1OpType::CX => Tk2Op::CX.into(),
            Tk1OpType::CY => Tk2Op::CY.into(),
            Tk1OpType::CZ => Tk2Op::CZ.into(),
            Tk1OpType::CRz => Tk2Op::CRz.into(),
            Tk1OpType::T => Tk2Op::T.into(),
            Tk1OpType::Tdg => Tk2Op::Tdg.into(),
            Tk1OpType::S => Tk2Op::S.into(),
            Tk1OpType::Sdg => Tk2Op::Sdg.into(),
            Tk1OpType::X => Tk2Op::X.into(),
            Tk1OpType::Y => Tk2Op::Y.into(),
            Tk1OpType::Z => Tk2Op::Z.into(),
            Tk1OpType::Rx => Tk2Op::Rx.into(),
            Tk1OpType::Ry => Tk2Op::Ry.into(),
            Tk1OpType::Rz => Tk2Op::Rz.into(),
            Tk1OpType::CCX => Tk2Op::Toffoli.into(),
            Tk1OpType::Reset => Tk2Op::Reset.into(),
            Tk1OpType::Measure => Tk2Op::Measure.into(),
            Tk1OpType::noop => Noop::new(qb_t()).into(),
            _ => {
                return None;
            }
        };
        Some(Self::new(op, Some(serial_op)))
    }

    /// Returns the dataflow signature for this operation.
    pub fn signature(&self) -> Option<Cow<'_, Signature>> {
        self.op.dataflow_signature()
    }

    /// Returns the tket2 optype for this operation.
    pub fn optype(&self) -> &OpType {
        &self.op
    }

    /// Consumes the `NativeOp` and returns the underlying `OpType`.
    pub fn into_optype(self) -> OpType {
        self.op
    }

    /// Returns the ports corresponding to parameters for this operation.
    pub fn param_ports(&self) -> impl Iterator<Item = IncomingPort> + '_ {
        self.signature().into_iter().flat_map(|sig| {
            let types = sig.input_types().to_owned();
            sig.input_ports()
                .zip(types)
                .filter(|(_, ty)| [rotation_type(), float64_type()].contains(ty))
                .map(|(port, _)| port)
        })
    }

    /// Update the internal bit/qubit/parameter counts.
    fn compute_counts(&mut self) {
        self.input_bits = 0;
        self.input_qubits = 0;
        self.num_params = 0;
        self.output_bits = 0;
        self.output_qubits = 0;
        let Some(sig) = self.signature().map(Cow::into_owned) else {
            return;
        };
        for ty in sig.input_types() {
            if ty == &qb_t() {
                self.input_qubits += 1;
            } else if ty == &bool_t() {
                self.input_bits += 1;
            } else if [rotation_type(), float64_type()].contains(ty) {
                self.num_params += 1;
            }
        }
        for ty in sig.output_types() {
            if ty == &qb_t() {
                self.output_qubits += 1;
            } else if ty == &bool_t() {
                self.output_bits += 1;
            }
        }
    }
}
