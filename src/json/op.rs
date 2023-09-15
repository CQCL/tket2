//! This module defines the internal `JsonOp` struct wrapping the logic for
//! going between `tket_json_rs::optype::OpType` and `hugr::ops::OpType`.
//!
//! The `JsonOp` tries to homogenize the
//! `tket_json_rs::circuit_json::Operation`s coming from the encoded TKET1
//! circuits by ensuring they always define a signature, and computing the
//! explicit count of qubits and linear bits.

use hugr::extension::prelude::QB_T;

use hugr::ops::custom::ExternalOp;
use hugr::ops::{LeafOp, OpTrait, OpType};
use hugr::std_extensions::arithmetic::float_types::FLOAT64_TYPE;
use hugr::types::FunctionType;

use itertools::Itertools;
use tket_json_rs::circuit_json;
use tket_json_rs::optype::OpType as JsonOpType;

use super::OpConvertError;
use crate::extension::{try_unwrap_json_op, LINEAR_BIT};
use crate::T2Op;

/// A serialized operation, containing the operation type and all its attributes.
///
/// Wrapper around [`circuit_json::Operation`] with cached number of qubits and bits.
///
/// The `Operation` contained by this struct is guaranteed to have a signature.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub(crate) struct JsonOp {
    op: circuit_json::Operation,
    num_qubits: usize,
    num_bits: usize,
    /// Node input for each parameter in `op.params`.
    ///
    /// If the input is `None`, the parameter does not use a Hugr port and is
    /// instead stored purely as metadata for the `Operation`.
    param_inputs: Vec<Option<usize>>,
    /// The number of non-None inputs in `param_inputs`, corresponding to the
    /// FLOAT64_TYPE inputs to the Hugr operation.
    num_params: usize,
}

impl JsonOp {
    /// Create a new `JsonOp` from a `circuit_json::Operation`, computing its
    /// number of qubits from the signature
    ///
    /// Fails if the operation does not define a signature. See
    /// [`JsonOp::new_from_op`] for a version that generates a signature if none
    /// is defined.
    #[allow(unused)]
    pub fn new(op: circuit_json::Operation) -> Option<Self> {
        let Some(sig) = &op.signature else {
            return None;
        };
        let input_counts = sig.iter().map(String::as_ref).counts();
        let num_qubits = input_counts.get("Q").copied().unwrap_or(0);
        let num_bits = input_counts.get("B").copied().unwrap_or(0);
        let mut op = Self {
            op,
            num_qubits,
            num_bits,
            param_inputs: Vec::new(),
            num_params: 0,
        };
        op.compute_param_fields();
        Some(op)
    }

    /// Create a new `JsonOp` from a `circuit_json::Operation`, with the number
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
            op,
            num_qubits,
            num_bits,
            param_inputs: Vec::new(),
            num_params: 0,
        };
        op.compute_param_fields();
        op
    }

    /// Create a new `JsonOp` from the optype and the number of parameters.
    pub fn new_with_counts(
        json_optype: JsonOpType,
        num_qubits: usize,
        num_bits: usize,
        num_params: usize,
    ) -> Self {
        let mut params = None;
        let mut param_inputs = vec![];
        if num_params > 0 {
            let offset = num_qubits + num_bits;
            params = Some(vec!["".into(); num_params]);
            param_inputs = (offset..offset + num_params).map(Option::Some).collect();
        }
        let op = circuit_json::Operation {
            op_type: json_optype,
            n_qb: Some(num_qubits as u32),
            params,
            op_box: None,
            signature: Some([vec!["Q".into(); num_qubits], vec!["B".into(); num_bits]].concat()),
            conditional: None,
        };
        Self {
            op,
            num_qubits,
            num_bits,
            param_inputs,
            num_params,
        }
    }

    /// Compute the signature of the operation.
    #[inline]
    pub fn signature(&self) -> FunctionType {
        let linear = [
            vec![QB_T; self.num_qubits],
            vec![LINEAR_BIT.clone(); self.num_bits],
        ]
        .concat();
        let params = vec![FLOAT64_TYPE; self.num_params];
        FunctionType::new([linear.clone(), params].concat(), linear)
        // .with_extension_delta(&ExtensionSet::singleton(&TKET1_EXTENSION_ID))
    }

    /// List of parameters in the operation that should be exposed as inputs.
    #[inline]
    pub fn param_inputs(&self) -> impl Iterator<Item = &str> {
        self.param_inputs
            .iter()
            .filter_map(|&i| self.op.params.as_ref()?.get(i?).map(String::as_ref))
    }

    pub fn into_operation(self) -> circuit_json::Operation {
        self.op
    }

    /// Wraps the op into a Hugr opaque operation
    fn as_opaque_op(&self) -> ExternalOp {
        crate::extension::wrap_json_op(self)
    }

    /// Compute the `parameter_input` and `num_params` fields by looking for
    /// parameters in `op.params` that can be mapped to input wires in the Hugr.
    ///
    /// Updates the internal `num_params` and `param_inputs` fields.
    fn compute_param_fields(&mut self) {
        let Some(params) = self.op.params.as_ref() else {
            self.param_inputs = vec![];
            self.num_params = 0;
            return;
        };

        self.num_params = params.len();
        self.param_inputs = (0..params.len()).map(Some).collect();
    }
}

impl From<&JsonOp> for OpType {
    /// Convert the operation into a HUGR operation.
    ///
    /// We only translate operations that have a 1:1 mapping between TKET and HUGR.
    /// Any other operation is wrapped in an `OpaqueOp`.
    fn from(json_op: &JsonOp) -> Self {
        match json_op.op.op_type {
            // JsonOpType::X => LeafOp::X.into(),
            JsonOpType::H => T2Op::H.into(),
            JsonOpType::CX => T2Op::CX.into(),
            JsonOpType::T => T2Op::T.into(),
            JsonOpType::Tdg => T2Op::Tdg.into(),
            JsonOpType::X => T2Op::X.into(),
            JsonOpType::Rz => T2Op::RzF64.into(),
            JsonOpType::Rx => T2Op::RxF64.into(),
            JsonOpType::TK1 => T2Op::TK1.into(),
            JsonOpType::PhasedX => T2Op::PhasedX.into(),
            JsonOpType::ZZMax => T2Op::ZZMax.into(),
            JsonOpType::ZZPhase => T2Op::ZZPhase.into(),
            JsonOpType::noop => LeafOp::Noop { ty: QB_T }.into(),
            _ => LeafOp::CustomOp(Box::new(json_op.as_opaque_op())).into(),
        }
    }
}

impl TryFrom<&OpType> for JsonOp {
    type Error = OpConvertError;

    fn try_from(op: &OpType) -> Result<Self, Self::Error> {
        // We only translate operations that have a 1:1 mapping between TKET and TKET2
        //
        // Other TKET1 operations are wrapped in an `OpaqueOp`.
        //
        // Non-supported Hugr operations throw an error.
        let err = || OpConvertError::UnsupportedOpSerialization(op.clone());
        let OpType::LeafOp(leaf) = op else {
            return Err(err());
        };

        let json_optype = if let Ok(t2op) = leaf.clone().try_into() {
            match t2op {
                T2Op::CX => JsonOpType::CX,
                T2Op::H => JsonOpType::H,
                T2Op::Measure => JsonOpType::Measure,
                T2Op::RzF64 => JsonOpType::Rz,
                T2Op::RxF64 => JsonOpType::Rx,
                // TODO: Use a TK2 opaque op once we update the tket-json-rs dependency.
                T2Op::AngleAdd => JsonOpType::AngleAdd,
                T2Op::TK1 => JsonOpType::TK1,
                T2Op::PhasedX => JsonOpType::PhasedX,
                T2Op::ZZMax => JsonOpType::ZZMax,
                T2Op::ZZPhase => JsonOpType::ZZPhase,
                _ => return Err(err()),
            }
        } else if let LeafOp::CustomOp(b) = leaf {
            let ext = (*b).as_ref();
            return try_unwrap_json_op(ext).ok_or_else(err);
        } else {
            return Err(err());
        };

        let mut num_qubits = 0;
        let mut num_bits = 0;
        let mut num_params = 0;
        for ty in op.signature().input.iter() {
            if ty == &QB_T {
                num_qubits += 1
            } else if *ty == *LINEAR_BIT {
                num_bits += 1
            } else if ty == &FLOAT64_TYPE {
                num_params += 1
            }
        }

        Ok(JsonOp::new_with_counts(
            json_optype,
            num_qubits,
            num_bits,
            num_params,
        ))
    }
}
