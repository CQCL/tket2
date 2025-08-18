//! Encoder/decoder for [qsystem::EXTENSION][use crate::extension::qsystem::EXTENSION] operations.

use std::sync::Arc;

use hugr::extension::simple_op::MakeExtensionOp;
use hugr::extension::ExtensionId;
use hugr::ops::ExtensionOp;
use hugr::HugrView;
use tket::serialize::pytket::decoder::{
    DecodeStatus, LoadedParameter, PytketDecoderContext, TrackedBit, TrackedQubit,
};
use tket::serialize::pytket::encoder::EncodeStatus;
use tket::serialize::pytket::extension::PytketDecoder;
use tket::serialize::pytket::{
    PytketDecodeError, PytketEmitter, PytketEncodeError, PytketEncoderContext,
};
use tket::Circuit;
use tket_json_rs::optype::OpType as PytketOptype;

use crate::extension;
use crate::extension::qsystem::{QSystemOp, RuntimeBarrierDef};

/// Encoder for [futures](crate::extension::futures) operations.
#[derive(Debug, Clone, Default)]
pub struct QSystemEmitter;

impl<H: HugrView> PytketEmitter<H> for QSystemEmitter {
    fn extensions(&self) -> Option<Vec<ExtensionId>> {
        Some(vec![extension::qsystem::EXTENSION_ID])
    }

    fn op_to_pytket(
        &self,
        node: H::Node,
        op: &ExtensionOp,
        circ: &Circuit<H>,
        encoder: &mut PytketEncoderContext<H>,
    ) -> Result<EncodeStatus, PytketEncodeError<H::Node>> {
        if let Ok(tket_op) = QSystemOp::from_extension_op(op) {
            self.encode_qsystem_op(node, tket_op, circ, encoder)
        } else if let Ok(sympy_op) = RuntimeBarrierDef::from_extension_op(op) {
            self.encode_runtime_barrier_op(node, sympy_op, circ, encoder)
        } else {
            Ok(EncodeStatus::Unsupported)
        }
    }
}

impl QSystemEmitter {
    /// Encode a tket operation into a pytket operation.
    fn encode_qsystem_op<H: HugrView>(
        &self,
        node: H::Node,
        qsystem_op: QSystemOp,
        circ: &Circuit<H>,
        encoder: &mut PytketEncoderContext<H>,
    ) -> Result<EncodeStatus, PytketEncodeError<H::Node>> {
        let serial_op = match qsystem_op {
            QSystemOp::Measure => PytketOptype::Measure,
            // "Lazy" operations are translated as eager measurements in pytket,
            // as there is no `Future<T>` type there.
            QSystemOp::LazyMeasure => PytketOptype::Measure,
            QSystemOp::Rz => PytketOptype::Rz,
            QSystemOp::PhasedX => PytketOptype::PhasedX,
            QSystemOp::ZZPhase => PytketOptype::ZZPhase,
            QSystemOp::Reset => PytketOptype::Reset,
            QSystemOp::QFree => {
                // Mark the qubit inputs as explored and forget about them.
                encoder.get_input_values(node, circ)?;
                return Ok(EncodeStatus::Success);
            }
            QSystemOp::LazyMeasureReset | QSystemOp::MeasureReset => {
                // These may require a pytket measurement followed by a reset.
                return Ok(EncodeStatus::Unsupported);
            }
            QSystemOp::LazyMeasureLeaked => {
                // No equivalent pytket operation.
                return Ok(EncodeStatus::Unsupported);
            }
            QSystemOp::TryQAlloc => {
                // Pytket circuits don't support the optional type returned by `TryQAlloc`.
                return Ok(EncodeStatus::Unsupported);
            }
        };

        // Most operations map directly to a pytket one.
        encoder.emit_node(serial_op, node, circ)?;

        Ok(EncodeStatus::Success)
    }

    fn encode_runtime_barrier_op<H: HugrView>(
        &self,
        node: H::Node,
        _runtime_barrier_op: RuntimeBarrierDef,
        circ: &Circuit<H>,
        encoder: &mut PytketEncoderContext<H>,
    ) -> Result<EncodeStatus, PytketEncodeError<H::Node>> {
        encoder.emit_node(PytketOptype::Barrier, node, circ)?;

        Ok(EncodeStatus::Success)
    }
}

impl PytketDecoder for QSystemEmitter {
    fn op_types(&self) -> Vec<PytketOptype> {
        // Process native optypes that are not supported by the `TketOp` emitter.
        vec![
            PytketOptype::PhasedX,
            PytketOptype::ZZPhase,
            PytketOptype::ZZMax,
        ]
    }

    fn op_to_hugr<'h>(
        &self,
        op: &tket_json_rs::circuit_json::Operation,
        qubits: &[TrackedQubit],
        bits: &[TrackedBit],
        params: &[Arc<LoadedParameter>],
        _opgroup: Option<&str>,
        decoder: &mut PytketDecoderContext<'h>,
    ) -> Result<DecodeStatus, PytketDecodeError> {
        let op = match op.op_type {
            PytketOptype::PhasedX => QSystemOp::PhasedX,
            PytketOptype::ZZPhase => QSystemOp::ZZPhase,
            PytketOptype::ZZMax => {
                // This is a ZZPhase with a 1/2 angle.
                let param = decoder.load_parameter("pi/2");
                decoder.add_node_with_wires(QSystemOp::ZZPhase, qubits, bits, &[param])?;
                return Ok(DecodeStatus::Success);
            }
            _ => {
                return Ok(DecodeStatus::Unsupported);
            }
        };
        decoder.add_node_with_wires(op, qubits, bits, params)?;

        Ok(DecodeStatus::Success)
    }
}
