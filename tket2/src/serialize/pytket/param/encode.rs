//! Definitions for encoding hugr graphs into pytket op parameters.

use hugr::ops::{OpType, Value};
use hugr::std_extensions::arithmetic::float_ops::FloatOps;
use hugr::std_extensions::arithmetic::float_types::ConstF64;

use crate::extension::rotation::{ConstRotation, RotationOp};
use crate::extension::sympy::SympyOp;
use crate::ops::match_symb_const_op;

/// Fold a rotation or float operation into a string, given the string
/// representations of its inputs.
///
/// The folded op must have a single string output.
///
/// Returns `None` if the operation cannot be folded.
pub fn fold_param_op(optype: &OpType, inputs: &[&str]) -> Option<String> {
    let param = match optype {
        OpType::Const(const_op) => {
            // New constant, register it if it can be interpreted as a parameter.
            try_constant_to_param(const_op.value())?
        }
        OpType::LoadConstant(_op_type) => {
            // Re-use the parameter from the input.
            inputs[0].to_string()
        }
        // Encode some angle and float operations directly as strings using
        // the already encoded inputs. Fail if the operation is not
        // supported, and let the operation encoding process it instead.
        OpType::ExtensionOp(_) => {
            if let Some(s) = optype
                .cast::<RotationOp>()
                .and_then(|op| encode_rotation_op(&op, inputs))
            {
                s
            } else if let Some(s) = optype
                .cast::<FloatOps>()
                .and_then(|op| encode_float_op(&op, inputs))
            {
                s
            } else if let Some(s) = optype
                .cast::<SympyOp>()
                .and_then(|op| encode_sympy_op(&op, inputs))
            {
                s
            } else {
                return None;
            }
        }
        _ => match_symb_const_op(optype)?.to_string(),
    };
    Some(param)
}

/// Convert a HUGR rotation or float constant to a TKET1 parameter.
///
/// Angle parameters in TKET1 are encoded as a number of half-turns,
/// whereas HUGR uses radians.
#[inline]
fn try_constant_to_param(val: &Value) -> Option<String> {
    if let Some(const_angle) = val.get_custom_value::<ConstRotation>() {
        let half_turns = const_angle.half_turns();
        Some(half_turns.to_string())
    } else if let Some(const_float) = val.get_custom_value::<ConstF64>() {
        let float = const_float.value();

        // Special case for pi rotations
        if float == std::f64::consts::PI {
            Some("pi".to_string())
        } else {
            Some(float.to_string())
        }
    } else {
        None
    }
}

/// Encode an [`RotationOp`]s as a string, given its encoded inputs.
///
/// `inputs` contains the expressions to compute each input.
fn encode_rotation_op(op: &RotationOp, inputs: &[&str]) -> Option<String> {
    let s = match op {
        RotationOp::radd => format!("({} + {})", inputs[0], inputs[1]),
        // Encode/decode the rotation as pytket parameters, expressed as half-turns.
        // Note that the tracked parameter strings are always written in half-turns,
        // so the conversion here is a no-op.
        RotationOp::to_halfturns => inputs[0].to_string(),
        RotationOp::from_halfturns_unchecked => inputs[0].to_string(),
        // The checked conversion returns an option, which we do not support.
        RotationOp::from_halfturns => return None,
    };
    Some(s)
}

/// Encode an [`FloatOps`] as a string, given its encoded inputs.
fn encode_float_op(op: &FloatOps, inputs: &[&str]) -> Option<String> {
    let s = match op {
        FloatOps::fadd => format!("({} + {})", inputs[0], inputs[1]),
        FloatOps::fsub => format!("({} - {})", inputs[0], inputs[1]),
        FloatOps::fneg => format!("(-{})", inputs[0]),
        FloatOps::fmul => format!("({} * {})", inputs[0], inputs[1]),
        FloatOps::fdiv => format!("({} / {})", inputs[0], inputs[1]),
        FloatOps::fpow => format!("({} ** {})", inputs[0], inputs[1]),
        FloatOps::ffloor => format!("floor({})", inputs[0]),
        FloatOps::fceil => format!("ceil({})", inputs[0]),
        FloatOps::fround => format!("round({})", inputs[0]),
        FloatOps::fmax => format!("max({}, {})", inputs[0], inputs[1]),
        FloatOps::fmin => format!("min({}, {})", inputs[0], inputs[1]),
        FloatOps::fabs => format!("abs({})", inputs[0]),
        _ => return None,
    };
    Some(s)
}

/// Encode a [`SympyOp`]s as a string.
fn encode_sympy_op(op: &SympyOp, inputs: &[&str]) -> Option<String> {
    if !inputs.is_empty() {
        return None;
    }

    Some(op.expr.clone())
}
