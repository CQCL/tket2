//! Encoder and decoder for tket2 operations with native pytket counterparts.

use super::PytketEmitter;
use crate::extension::sympy::SympyOp;
use crate::extension::TKET2_EXTENSION_ID;
use crate::serialize::pytket::encoder::{EncodeStatus, Tk1EncoderContext};
use crate::serialize::pytket::Tk1ConvertError;
use crate::{Circuit, Tk2Op};
use hugr::extension::simple_op::MakeExtensionOp;
use hugr::extension::ExtensionId;
use hugr::ops::ExtensionOp;
use hugr::{HugrView, Wire};
use tket_json_rs::optype::OpType as Tk1OpType;

/// Encoder for [Tk2Op] operations.
#[derive(Debug, Clone, Default)]
pub struct Tk2Emitter;

impl<H: HugrView> PytketEmitter<H> for Tk2Emitter {
    fn extensions(&self) -> Option<Vec<ExtensionId>> {
        Some(vec![TKET2_EXTENSION_ID])
    }

    fn op_to_pytket(
        &self,
        node: H::Node,
        op: &ExtensionOp,
        circ: &Circuit<H>,
        encoder: &mut Tk1EncoderContext<H>,
    ) -> Result<EncodeStatus, Tk1ConvertError<H::Node>> {
        if let Ok(tk2op) = Tk2Op::from_extension_op(op) {
            self.encode_tk2_op(node, tk2op, circ, encoder)
        } else if let Ok(sympy_op) = SympyOp::from_extension_op(op) {
            self.encode_sympy_op(node, sympy_op, circ, encoder)
        } else {
            Ok(EncodeStatus::Unsupported)
        }
    }
}

impl Tk2Emitter {
    /// Encode a tket2 operation into a pytket operation.
    fn encode_tk2_op<H: HugrView>(
        &self,
        node: H::Node,
        tk2op: Tk2Op,
        circ: &Circuit<H>,
        encoder: &mut Tk1EncoderContext<H>,
    ) -> Result<EncodeStatus, Tk1ConvertError<H::Node>> {
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
            Tk2Op::V => Tk1OpType::V,
            Tk2Op::Vdg => Tk1OpType::Vdg,
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
                encoder.values.register_wire(wire, [qb], circ)?;
                return Ok(EncodeStatus::Success);
            }
            // Since the qubit still gets connected at the end of the circuit,
            // `QFree` is a no-op.
            Tk2Op::QFree => {
                return Ok(EncodeStatus::Success);
            }
            // Unsupported
            Tk2Op::TryQAlloc => {
                return Ok(EncodeStatus::Unsupported);
            }
        };

        // Most operations map directly to a pytket one.
        encoder.emit_node(serial_op, node, circ)?;

        Ok(EncodeStatus::Success)
    }

    /// Encode a tket2 sympy operation into a pytket operation.
    fn encode_sympy_op<H: HugrView>(
        &self,
        node: H::Node,
        sympy_op: SympyOp,
        circ: &Circuit<H>,
        encoder: &mut Tk1EncoderContext<H>,
    ) -> Result<EncodeStatus, Tk1ConvertError<H::Node>> {
        encoder.emit_transparent_node(node, circ, |_| vec![sympy_op.expr.clone()])?;
        Ok(EncodeStatus::Success)
    }
}
