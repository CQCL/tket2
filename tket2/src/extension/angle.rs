use std::{cmp::max, num::NonZeroU64};

use hugr::{
    extension::{prelude::ERROR_TYPE, ExtensionRegistry, SignatureError, TypeDef, PRELUDE},
    types::{
        type_param::{TypeArgError, TypeParam},
        ConstTypeError, CustomCheckFailure, CustomType, FunctionType, PolyFuncType, Type, TypeArg,
        TypeBound,
    },
    values::CustomConst,
    Extension,
};
use itertools::Itertools;
use smol_str::SmolStr;
use std::f64::consts::TAU;

/// Identifier for the angle type.
const ANGLE_TYPE_ID: SmolStr = SmolStr::new_inline("angle");

pub(super) fn angle_custom_type(extension: &Extension, log_denom_arg: TypeArg) -> CustomType {
    angle_def(extension).instantiate([log_denom_arg]).unwrap()
}

/// The largest permitted log-denominator.
pub const LOG_DENOM_MAX: u8 = 53;

const fn is_valid_log_denom(n: u8) -> bool {
    n <= LOG_DENOM_MAX
}

/// Type parameter for the log-denominator of an angle.
pub const LOG_DENOM_TYPE_PARAM: TypeParam =
    TypeParam::bounded_nat(NonZeroU64::MIN.saturating_add(LOG_DENOM_MAX as u64));

/// Get the log-denominator of the specified type argument or error if the argument is invalid.
fn get_log_denom(arg: &TypeArg) -> Result<u8, TypeArgError> {
    match arg {
        TypeArg::BoundedNat { n } if is_valid_log_denom(*n as u8) => Ok(*n as u8),
        _ => Err(TypeArgError::TypeMismatch {
            arg: arg.clone(),
            param: LOG_DENOM_TYPE_PARAM,
        }),
    }
}

pub(super) const fn type_arg(log_denom: u8) -> TypeArg {
    TypeArg::BoundedNat {
        n: log_denom as u64,
    }
}

/// An angle
#[derive(Clone, Debug, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ConstAngle {
    log_denom: u8,
    value: u64,
}

impl ConstAngle {
    /// Create a new [`ConstAngle`] from a log-denominator and a numerator
    pub fn new(log_denom: u8, value: u64) -> Result<Self, ConstTypeError> {
        if !is_valid_log_denom(log_denom) {
            return Err(ConstTypeError::CustomCheckFail(
                hugr::types::CustomCheckFailure::Message(
                    "Invalid angle log-denominator.".to_owned(),
                ),
            ));
        }
        if value >= (1u64 << log_denom) {
            return Err(ConstTypeError::CustomCheckFail(
                hugr::types::CustomCheckFailure::Message(
                    "Invalid unsigned integer value.".to_owned(),
                ),
            ));
        }
        Ok(Self { log_denom, value })
    }

    /// Create a new [`ConstAngle`] from a log-denominator and a floating-point value in radians,
    /// rounding to the nearest corresponding value. (Ties round away from zero.)
    pub fn from_radians_rounding(log_denom: u8, theta: f64) -> Result<Self, ConstTypeError> {
        if !is_valid_log_denom(log_denom) {
            return Err(ConstTypeError::CustomCheckFail(
                hugr::types::CustomCheckFailure::Message(
                    "Invalid angle log-denominator.".to_owned(),
                ),
            ));
        }
        let a = (((1u64 << log_denom) as f64) * theta / TAU).round() as i64;
        Ok(Self {
            log_denom,
            value: a.rem_euclid(1i64 << log_denom) as u64,
        })
    }

    /// Returns the value of the constant
    pub fn value(&self) -> u64 {
        self.value
    }

    /// Returns the log-denominator of the constant
    pub fn log_denom(&self) -> u8 {
        self.log_denom
    }
}

#[typetag::serde]
impl CustomConst for ConstAngle {
    fn name(&self) -> SmolStr {
        format!("a(2π*{}/2^{})", self.value, self.log_denom).into()
    }
    fn check_custom_type(&self, typ: &CustomType) -> Result<(), CustomCheckFailure> {
        if typ.clone() == super::angle_custom_type(self.log_denom) {
            Ok(())
        } else {
            Err(CustomCheckFailure::Message(
                "Angle constant type mismatch.".into(),
            ))
        }
    }
    fn equal_consts(&self, other: &dyn CustomConst) -> bool {
        hugr::values::downcast_equal_consts(self, other)
    }
}

fn type_var(var_id: usize, extension: &Extension) -> Result<Type, SignatureError> {
    Ok(Type::new_extension(angle_def(extension).instantiate(
        vec![TypeArg::new_var_use(var_id, LOG_DENOM_TYPE_PARAM)],
    )?))
}
fn atrunc_sig(extension: &Extension) -> Result<FunctionType, SignatureError> {
    let in_angle = type_var(0, extension)?;
    let out_angle = type_var(1, extension)?;

    Ok(FunctionType::new(vec![in_angle], vec![out_angle]))
}

fn aconvert_sig(extension: &Extension) -> Result<FunctionType, SignatureError> {
    let in_angle = type_var(0, extension)?;
    let out_angle = type_var(1, extension)?;
    Ok(FunctionType::new(
        vec![in_angle],
        vec![Type::new_sum(vec![out_angle, ERROR_TYPE])],
    ))
}

/// Collect a vector into an array.
fn collect_array<const N: usize, T: std::fmt::Debug>(arr: &[T]) -> [&T; N] {
    arr.iter().collect_vec().try_into().unwrap()
}

fn abinop_sig(arg_values: &[TypeArg]) -> Result<FunctionType, SignatureError> {
    let [arg0, arg1] = collect_array(arg_values);
    let m: u8 = get_log_denom(arg0)?;
    let n: u8 = get_log_denom(arg1)?;
    let l: u8 = max(m, n);
    let ang_typ = |n| Type::new_extension(super::angle_custom_type(n));
    Ok(FunctionType::new(
        vec![ang_typ(n), ang_typ(m)],
        vec![ang_typ(l)],
    ))
}

fn aunop_sig(extension: &Extension) -> Result<FunctionType, SignatureError> {
    let angle = type_var(0, extension)?;
    Ok(FunctionType::new_linear(vec![angle]))
}

fn angle_def(extension: &Extension) -> &TypeDef {
    extension.get_type(&ANGLE_TYPE_ID).unwrap()
}

pub(super) fn add_to_extension(extension: &mut Extension) {
    extension
        .add_type(
            ANGLE_TYPE_ID,
            vec![LOG_DENOM_TYPE_PARAM],
            "angle value with a given log-denominator".to_owned(),
            TypeBound::Eq.into(),
        )
        .unwrap();

    let reg1: ExtensionRegistry = [PRELUDE.to_owned(), extension.to_owned()].into();
    extension
        .add_op_type_scheme(
            "atrunc".into(),
            "truncate an angle to one with a lower log-denominator with the same value, rounding \
            down in [0, 2π) if necessary"
                .to_owned(),
            Default::default(),
            vec![],
            PolyFuncType::new_validated(
                vec![LOG_DENOM_TYPE_PARAM, LOG_DENOM_TYPE_PARAM],
                atrunc_sig(extension).unwrap(),
                &reg1,
            )
            .unwrap(),
        )
        .unwrap();

    extension
        .add_op_type_scheme(
            "aconvert".into(),
            "convert an angle to one with another log-denominator having the same value, if \
            possible, otherwise return an error"
                .to_owned(),
            Default::default(),
            vec![],
            PolyFuncType::new_validated(
                vec![LOG_DENOM_TYPE_PARAM, LOG_DENOM_TYPE_PARAM],
                aconvert_sig(extension).unwrap(),
                &reg1,
            )
            .unwrap(),
        )
        .unwrap();

    extension
        .add_op_custom_sig_simple(
            "aadd".into(),
            "addition of angles".to_owned(),
            vec![LOG_DENOM_TYPE_PARAM],
            abinop_sig,
        )
        .unwrap();

    extension
        .add_op_custom_sig_simple(
            "asub".into(),
            "subtraction of the second angle from the first".to_owned(),
            vec![LOG_DENOM_TYPE_PARAM],
            abinop_sig,
        )
        .unwrap();

    extension
        .add_op_type_scheme(
            "aneg".into(),
            "negation of an angle".to_owned(),
            Default::default(),
            vec![],
            PolyFuncType::new_validated(
                vec![LOG_DENOM_TYPE_PARAM, LOG_DENOM_TYPE_PARAM],
                aunop_sig(extension).unwrap(),
                &reg1,
            )
            .unwrap(),
        )
        .unwrap();
}

#[cfg(test)]
mod test {
    use super::*;
    use hugr::types::TypeArg;

    #[test]
    fn test_angle_log_denoms() {
        let type_arg_53 = TypeArg::BoundedNat { n: 53 };
        assert_eq!(get_log_denom(&type_arg_53).unwrap(), 53);

        let type_arg_54 = TypeArg::BoundedNat { n: 54 };
        assert!(matches!(
            get_log_denom(&type_arg_54),
            Err(TypeArgError::TypeMismatch { .. })
        ));
    }

    #[test]
    fn test_angle_consts() {
        let const_a32_7 = ConstAngle::new(5, 7).unwrap();
        let const_a33_7 = ConstAngle::new(6, 7).unwrap();
        let const_a32_8 = ConstAngle::new(6, 8).unwrap();
        assert_ne!(const_a32_7, const_a33_7);
        assert_ne!(const_a32_7, const_a32_8);
        assert_eq!(const_a32_7, ConstAngle::new(5, 7).unwrap());
        assert!(matches!(
            ConstAngle::new(3, 256),
            Err(ConstTypeError::CustomCheckFail(_))
        ));
        assert!(matches!(
            ConstAngle::new(54, 256),
            Err(ConstTypeError::CustomCheckFail(_))
        ));
        let const_af1 = ConstAngle::from_radians_rounding(5, 0.21874 * TAU).unwrap();
        assert_eq!(const_af1.value(), 7);
        assert_eq!(const_af1.log_denom(), 5);
    }
}
