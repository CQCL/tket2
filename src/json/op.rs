//! This module defines the internal `JsonOp` struct wrapping the logic for
//! going between `tket_json_rs::optype::OpType` and `hugr::ops::OpType`.
//!
//! The `JsonOp` tries to homogenize the
//! `tket_json_rs::circuit_json::Operation`s coming from the encoded TKET1
//! circuits by ensuring they always define a signature, and computing the
//! explicit count of qubits and linear bits.

use hugr::extension::prelude::QB_T;
use hugr::extension::ExtensionSet;
use hugr::ops::custom::ExternalOp;
use hugr::ops::{LeafOp, OpTrait, OpType};
use hugr::std_extensions::arithmetic::float_types::FLOAT64_TYPE;
use hugr::types::FunctionType;

use itertools::Itertools;
use tket_json_rs::circuit_json;
use tket_json_rs::optype::OpType as JsonOpType;

use super::{try_param_to_constant, OpConvertError};
use crate::extension::{try_unwrap_json_op, LINEAR_BIT, TKET1_EXTENSION_ID};
use crate::utils::{cx_gate, h_gate};

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
        let Some(sig) = &op.signature else { return None; };
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
            .with_extension_delta(&ExtensionSet::singleton(&TKET1_EXTENSION_ID))
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

        let mut p_input_indices = 0..;
        let param_inputs = params
            .iter()
            .map(|param| try_param_to_constant(param).map(|_| p_input_indices.next().unwrap()))
            .collect();

        self.num_params = p_input_indices.next().unwrap();
        self.param_inputs = param_inputs;
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
            JsonOpType::H => h_gate().into(),
            JsonOpType::CX => cx_gate().into(),
            JsonOpType::noop => LeafOp::Noop { ty: QB_T }.into(),
            // TODO TKET1 measure takes a bit as input, HUGR measure does not
            //JsonOpType::Measure => LeafOp::Measure.into(),
            // JsonOpType::Reset => LeafOp::Reset.into(),
            // JsonOpType::ZZMax => LeafOp::ZZMax.into(),
            // JsonOpType::Rz => LeafOp::RzF64.into(),
            // JsonOpType::RzF64 => LeafOp::RzF64.into(),
            // TODO TKET1 I/O needs some special handling
            //JsonOpType::Input => hugr::ops::Input {
            //    types: json_op.signature().output,
            //    extensions: Default::default(),
            //}
            //.into(),
            //JsonOpType::Output => hugr::ops::Output {
            //    types: json_op.signature().input,
            //    extensions: Default::default(),
            //}
            //.into(),
            // JsonOpType::Z => LeafOp::Z.into(),
            // JsonOpType::Y => LeafOp::Y.into(),
            // JsonOpType::S => LeafOp::S.into(),
            // JsonOpType::Sdg => LeafOp::Sadj.into(),
            // JsonOpType::T => LeafOp::T.into(),
            // JsonOpType::Tdg => LeafOp::Tadj.into(),
            _ => LeafOp::CustomOp(Box::new(json_op.as_opaque_op())).into(),
        }
    }
}

impl TryFrom<&OpType> for JsonOp {
    type Error = OpConvertError;

    fn try_from(op: &OpType) -> Result<Self, Self::Error> {
        // We only translate operations that have a 1:1 mapping between TKET and HUGR
        //
        // Other TKET1 operations are wrapped in an `OpaqueOp`.
        //
        // Non-supported Hugr operations throw an error.
        let err = || OpConvertError::UnsupportedOpSerialization(op.clone());

        if let OpType::LeafOp(LeafOp::CustomOp(ext)) = op {
            return try_unwrap_json_op(ext).ok_or_else(err);
        }

        let json_optype: JsonOpType = match op {
            OpType::LeafOp(LeafOp::Noop { .. }) => JsonOpType::noop,
            OpType::LeafOp(LeafOp::CustomOp(b)) => match (*b).as_ref() {
                ExternalOp::Extension(c) if c.def().extension() == "quantum" => {
                    match &c.def().name()[..] {
                        "H" => JsonOpType::H,
                        "CX" => JsonOpType::CX,
                        _ => return Err(err()),
                    }
                }
                _ => return Err(err()),
                // h_gate() => JsonOpType::H,
                // cx_gate() => JsonOpType::CX,
                // LeafOp::ZZMax => JsonOpType::ZZMax,
                // LeafOp::Reset => JsonOpType::Reset,
                // //LeafOp::Measure => JsonOpType::Measure,
                // LeafOp::T => JsonOpType::T,
                // LeafOp::S => JsonOpType::S,
                // LeafOp::X => JsonOpType::X,
                // LeafOp::Y => JsonOpType::Y,
                // LeafOp::Z => JsonOpType::Z,
                // LeafOp::Tadj => JsonOpType::Tdg,
                // LeafOp::Sadj => JsonOpType::Sdg,
                //LeafOp::RzF64 => JsonOpType::Rz, // The angle in RzF64 comes from a constant input
                //LeafOp::Xor => todo!(),
                //LeafOp::MakeTuple { .. } => todo!(),
                //LeafOp::UnpackTuple { .. } => todo!(),
                //LeafOp::Tag { .. } => todo!(),
                //LeafOp::Lift { .. } => todo!(),
                // CustomOp is handled above
            },
            //OpType::Input(_) => JsonOpType::Input,
            //OpType::Output(_) => JsonOpType::Output,
            //hugr::ops::OpType::FuncDefn(_) => todo!(),
            //hugr::ops::OpType::FuncDecl(_) => todo!(),
            //hugr::ops::OpType::Const(_) => todo!(),
            //hugr::ops::OpType::Call(_) => todo!(),
            //hugr::ops::OpType::CallIndirect(_) => todo!(),
            //hugr::ops::OpType::LoadConstant(_) => todo!(),
            //hugr::ops::OpType::DFG(_) => JsonOpType::CircBox, // TODO: Requires generating the Operation::op_box
            _ => return Err(err()),
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

        if num_params > 0 {
            unimplemented!("Native parametric operation encoding is not supported yet.")
            // TODO: Gather parameter values from the `OpType` to encode in the `JsonOpType`.
        }

        Ok(JsonOp::new_with_counts(
            json_optype,
            num_qubits,
            num_bits,
            num_params,
        ))
    }
}
