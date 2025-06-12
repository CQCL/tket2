//! Encoder for pytket operations that cannot be represented naturally in tket2.

use crate::extension::rotation::rotation_type;
use crate::extension::{TKET1_EXTENSION, TKET1_EXTENSION_ID, TKET1_OP_NAME};
use crate::serialize::pytket::encoder::EncodeStatus;
use crate::serialize::pytket::{Tk1ConvertError, Tk1EncoderContext};
use crate::Circuit;

use super::PytketEmitter;
use hugr::extension::prelude::{bool_t, qb_t};
use hugr::extension::ExtensionId;
use hugr::ops::ExtensionOp;
use hugr::types::{Signature, TypeArg};
use hugr::{HugrView, IncomingPort};
use tket_json_rs::circuit_json;

/// Encoder for [TKET1_EXTENSION] operations.
///
/// That is, operations originating from a pytket circuit that did not have a
/// native HUGR representation and were instead serialized as opaque black-box
/// operations.
#[derive(Debug, Clone, Default)]
pub struct Tk1Emitter;

impl<H: HugrView> PytketEmitter<H> for Tk1Emitter {
    fn extensions(&self) -> Option<Vec<ExtensionId>> {
        Some(vec![TKET1_EXTENSION_ID])
    }

    fn op_to_pytket(
        &self,
        node: H::Node,
        op: &ExtensionOp,
        circ: &Circuit<H>,
        encoder: &mut Tk1EncoderContext<H>,
    ) -> Result<EncodeStatus, Tk1ConvertError<H::Node>> {
        if op.qualified_id() != format!("{TKET1_EXTENSION_ID}.{TKET1_OP_NAME}") {
            return Ok(EncodeStatus::Unsupported);
        }
        let Some(TypeArg::String { arg }) = op.args().first() else {
            return Err(Tk1ConvertError::custom(
                "Opaque TKET1 operation did not have a json-encoded type argument.",
            ));
        };
        let op: OpaqueTk1Op = serde_json::from_str(arg)?;

        // Most operations map directly to a pytket one.
        encoder.emit_node_command(
            node,
            circ,
            // We don't support opaque pytket operations with parameter outputs.
            |_args| Vec::new(),
            // Emit the pre-defined pytket operation stored in the metadata.
            move |_| op.serialised_op,
        )?;

        Ok(EncodeStatus::Success)
    }
}

/// A serialized operation, containing the operation type and all its attributes.
///
/// This value is only used if the operation does not have a native TKET2
/// counterpart that can be represented as a [`NativeOp`].
///
/// Wrapper around [`tket_json_rs::circuit_json::Operation`] with cached number of qubits and bits.
///
/// The `Operation` contained by this struct is guaranteed to have a signature.
///
///   [`NativeOp`]: super::native::NativeOp
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct OpaqueTk1Op {
    /// Internal operation data.
    #[serde(rename = "op")]
    pub serialised_op: circuit_json::Operation,
    /// Number of qubits declared by the operation.
    pub num_qubits: usize,
    /// Number of bits declared by the operation.
    pub num_bits: usize,
    /// Node input for each parameter in `op.params`.
    ///
    /// If the input is `None`, the parameter does not use a Hugr port and is
    /// instead stored purely as metadata for the `Operation`.
    pub param_inputs: Vec<Option<IncomingPort>>,
    /// The number of non-None inputs in `param_inputs`, corresponding to the
    /// rotation_type() inputs to the Hugr operation.
    pub num_params: usize,
}

impl OpaqueTk1Op {
    /// Create a new `OpaqueTk1Op` from a `circuit_json::Operation`, with the number
    /// of qubits and bits explicitly specified.
    ///
    /// If the operation does not define a signature, one is generated with the
    /// given amounts.
    pub fn new_from_op(
        mut op: circuit_json::Operation,
        num_qubits: usize,
        num_bits: usize,
    ) -> Self {
        if op.signature.is_none() {
            op.signature =
                Some([vec!["Q".into(); num_qubits], vec!["B".into(); num_bits]].concat());
        }
        let mut op = Self {
            serialised_op: op,
            num_qubits,
            num_bits,
            param_inputs: Vec::new(),
            num_params: 0,
        };
        op.compute_param_fields();
        op
    }

    /// Compute the signature of the operation.
    ///
    /// The signature returned has `num_qubits` qubit inputs, followed by
    /// `num_bits` bit inputs, followed by `num_params` `f64` inputs. It has
    /// `num_qubits` qubit outputs followed by `num_bits` bit outputs.
    #[inline]
    pub fn signature(&self) -> Signature {
        let linear = [
            vec![qb_t(); self.num_qubits],
            vec![bool_t().clone(); self.num_bits],
        ]
        .concat();
        let params = vec![rotation_type(); self.num_params];
        Signature::new([linear.clone(), params].concat(), linear)
    }

    /// Returns the ports corresponding to parameters for this operation.
    pub fn param_ports(&self) -> impl Iterator<Item = IncomingPort> + '_ {
        self.param_inputs.iter().filter_map(|&i| i)
    }

    /// Wraps the op into a [`TKET1_OP_NAME`] opaque operation.
    pub fn as_extension_op(&self) -> ExtensionOp {
        let payload = TypeArg::String {
            arg: serde_json::to_string(self).unwrap(),
        };
        let op_def = TKET1_EXTENSION.get_op(&TKET1_OP_NAME).unwrap();
        ExtensionOp::new(op_def.clone(), vec![payload]).unwrap_or_else(|e| panic!("{e}"))
    }

    /// Compute the `parameter_input` and `num_params` fields by looking for
    /// parameters in `op.params` that can be mapped to input wires in the Hugr.
    ///
    /// Updates the internal `num_params` and `param_inputs` fields.
    fn compute_param_fields(&mut self) {
        let Some(params) = self.serialised_op.params.as_ref() else {
            self.param_inputs = vec![];
            self.num_params = 0;
            return;
        };

        self.num_params = params.len();
        self.param_inputs = (0..params.len()).map(|i| Some(i.into())).collect();
    }
}
