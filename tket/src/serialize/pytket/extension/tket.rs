//! Encoder and decoder for tket operations with native pytket counterparts.

use super::PytketEmitter;
use crate::extension::sympy::SympyOp;
use crate::extension::TKET_EXTENSION_ID;
use crate::serialize::pytket::encoder::{EncodeStatus, PytketEncoderContext};
use crate::serialize::pytket::PytketEncodeError;
use crate::{Circuit, TketOp};
use hugr::extension::simple_op::MakeExtensionOp;
use hugr::extension::ExtensionId;
use hugr::ops::ExtensionOp;
use hugr::{HugrView, Wire};
use tket_json_rs::optype::OpType as Tk1OpType;

/// Encoder for [TketOp] operations.
#[derive(Debug, Clone, Default)]
pub struct TketOpEmitter;

impl<H: HugrView> PytketEmitter<H> for TketOpEmitter {
    fn extensions(&self) -> Option<Vec<ExtensionId>> {
        Some(vec![TKET_EXTENSION_ID])
    }

    fn op_to_pytket(
        &self,
        node: H::Node,
        op: &ExtensionOp,
        circ: &Circuit<H>,
        encoder: &mut PytketEncoderContext<H>,
    ) -> Result<EncodeStatus, PytketEncodeError<H::Node>> {
        if let Ok(tket_op) = TketOp::from_extension_op(op) {
            self.encode_tket_op(node, tket_op, circ, encoder)
        } else if let Ok(sympy_op) = SympyOp::from_extension_op(op) {
            self.encode_sympy_op(node, sympy_op, circ, encoder)
        } else {
            Ok(EncodeStatus::Unsupported)
        }
    }
}

impl TketOpEmitter {
    /// Encode a tket operation into a pytket operation.
    fn encode_tket_op<H: HugrView>(
        &self,
        node: H::Node,
        tket_op: TketOp,
        circ: &Circuit<H>,
        encoder: &mut PytketEncoderContext<H>,
    ) -> Result<EncodeStatus, PytketEncodeError<H::Node>> {
        let serial_op = match tket_op {
            TketOp::H => Tk1OpType::H,
            TketOp::CX => Tk1OpType::CX,
            TketOp::CY => Tk1OpType::CY,
            TketOp::CZ => Tk1OpType::CZ,
            TketOp::CRz => Tk1OpType::CRz,
            TketOp::T => Tk1OpType::T,
            TketOp::Tdg => Tk1OpType::Tdg,
            TketOp::S => Tk1OpType::S,
            TketOp::Sdg => Tk1OpType::Sdg,
            TketOp::V => Tk1OpType::V,
            TketOp::Vdg => Tk1OpType::Vdg,
            TketOp::X => Tk1OpType::X,
            TketOp::Y => Tk1OpType::Y,
            TketOp::Z => Tk1OpType::Z,
            TketOp::Rx => Tk1OpType::Rx,
            TketOp::Rz => Tk1OpType::Rz,
            TketOp::Ry => Tk1OpType::Ry,
            TketOp::Toffoli => Tk1OpType::CCX,
            TketOp::Reset => Tk1OpType::Reset,
            TketOp::Measure => Tk1OpType::Measure,
            // We translate `MeasureFree` the same way as a `Measure` operation.
            // Since the node does not have outputs the qubit/bit will simply be ignored,
            // but will appear when collecting the final pytket registers.
            TketOp::MeasureFree => Tk1OpType::Measure,
            // These operations are implicitly supported by the encoding,
            // they do not create a new command but just modify the value trackers.
            TketOp::QAlloc => {
                let out_port = circ.hugr().node_outputs(node).next().unwrap();
                let wire = Wire::new(node, out_port);
                let qb = encoder.values.new_qubit();
                encoder.values.register_wire(wire, [qb], circ)?;
                return Ok(EncodeStatus::Success);
            }
            // Since the qubit still gets connected at the end of the circuit,
            // `QFree` is a no-op.
            TketOp::QFree => {
                return Ok(EncodeStatus::Success);
            }
            // Unsupported
            TketOp::TryQAlloc => {
                return Ok(EncodeStatus::Unsupported);
            }
        };

        // Most operations map directly to a pytket one.
        encoder.emit_node(serial_op, node, circ)?;

        Ok(EncodeStatus::Success)
    }

    /// Encode a tket sympy operation into a pytket operation.
    fn encode_sympy_op<H: HugrView>(
        &self,
        node: H::Node,
        sympy_op: SympyOp,
        circ: &Circuit<H>,
        encoder: &mut PytketEncoderContext<H>,
    ) -> Result<EncodeStatus, PytketEncodeError<H::Node>> {
        encoder.emit_transparent_node(node, circ, |_| vec![sympy_op.expr.clone()])?;
        Ok(EncodeStatus::Success)
    }
}
