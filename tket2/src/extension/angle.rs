use hugr::extension::prelude::{sum_with_error, BOOL_T, USIZE_T};
use hugr::extension::simple_op::{MakeOpDef, MakeRegisteredOp};
use hugr::extension::{ExtensionId, ExtensionSet, Version};
use hugr::ops::constant::{downcast_equal_consts, CustomConst};
use hugr::std_extensions::arithmetic::float_types::FLOAT64_TYPE;
use hugr::type_row;
use hugr::{
    types::{ConstTypeError, CustomType, Signature, Type, TypeBound},
    Extension,
};
use smol_str::SmolStr;
use std::f64::consts::TAU;
use strum::{EnumIter, EnumString, IntoStaticStr};

use lazy_static::lazy_static;

/// Name of tket 2 angle extension.
pub const ANGLE_EXTENSION_ID: ExtensionId = ExtensionId::new_unchecked("tket2.angle");

/// Current version of the TKET 2 angle extension
pub const ANGLE_EXTENSION_VERSION: Version = Version::new(0, 1, 0);

lazy_static! {
    /// The extension definition for TKET2 angle type and ops.
    pub static ref ANGLE_EXTENSION: Extension = {
        let mut e = Extension::new(ANGLE_EXTENSION_ID, ANGLE_EXTENSION_VERSION);
        add_to_extension(&mut e);
        e
    };
}

/// Identifier for the angle type.
const ANGLE_TYPE_ID: SmolStr = SmolStr::new_inline("angle");
/// Dyadic rational angle type (as [CustomType])
pub const ANGLE_CUSTOM_TYPE: CustomType =
    CustomType::new_simple(ANGLE_TYPE_ID, ANGLE_EXTENSION_ID, TypeBound::Copyable);

/// Type representing an angle that is a dyadic rational multiple of π (as [Type])
pub const ANGLE_TYPE: Type = Type::new_extension(ANGLE_CUSTOM_TYPE);

/// The largest permitted log-denominator.
pub const LOG_DENOM_MAX: u8 = 53;

const fn is_valid_log_denom(n: u8) -> bool {
    n <= LOG_DENOM_MAX
}

/// An angle
#[derive(Clone, Debug, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ConstAngle {
    log_denom: u8,
    value: u64,
}

impl ConstAngle {
    /// The constant π
    pub const PI: Self = Self::new_unchecked(1, 1);
    /// The constant 2π
    pub const TAU: Self = Self::new_unchecked(0, 1);
    /// The constant π/2
    pub const PI_2: Self = Self::new_unchecked(2, 1);
    /// The constant π/4
    pub const PI_4: Self = Self::new_unchecked(3, 1);

    /// Create a new [`ConstAngle`] from a log-denominator and a numerator without
    /// checking for validity.
    const fn new_unchecked(log_denom: u8, value: u64) -> Self {
        Self { log_denom, value }
    }
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

    /// Create a new [`ConstAngle`] from a floating-point value in radians,
    /// using the highest possible log-denominator and
    /// rounding to the nearest corresponding value. (Ties round away from zero.)
    pub fn from_radians_rounding_max(theta: f64) -> Result<Self, ConstTypeError> {
        Self::from_radians_rounding(LOG_DENOM_MAX, theta)
    }

    /// Returns the value of the constant
    pub fn value(&self) -> u64 {
        self.value
    }

    /// Returns the log-denominator of the constant
    pub fn log_denom(&self) -> u8 {
        self.log_denom
    }

    /// Returns the value of the constant in radians
    pub fn to_radians(&self) -> f64 {
        self.to_turns() * TAU
    }

    /// Returns the value of the constant divided by 2π
    pub fn to_turns(&self) -> f64 {
        (self.value as f64) / (1u64 << self.log_denom) as f64
    }
}

#[typetag::serde]
impl CustomConst for ConstAngle {
    fn name(&self) -> SmolStr {
        format!("a(2π*{}/2^{})", self.value, self.log_denom).into()
    }

    fn get_type(&self) -> Type {
        ANGLE_TYPE
    }

    fn equal_consts(&self, other: &dyn CustomConst) -> bool {
        downcast_equal_consts(self, other)
    }
    fn extension_reqs(&self) -> ExtensionSet {
        ExtensionSet::singleton(&ANGLE_EXTENSION_ID)
    }
}

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq, EnumIter, IntoStaticStr, EnumString)]
#[allow(missing_docs, non_camel_case_types)]
#[non_exhaustive]
/// Angle operations
pub enum AngleOp {
    /// Truncate an angle to one with a lower log-denominator with the nearest value, rounding down in [0, 2π) if necessary
    atrunc,
    /// Addition of angles
    aadd,
    /// Subtraction of the second angle from the first
    asub,
    /// Negation of an angle
    aneg,
    /// Construct angle from numerator and log-denominator
    anew,
    /// Decompose angle into numerator and log-denominator
    aparts,
    /// Construct angle from radians
    afromrad,
    /// Convert angle to radians
    atorad,
    /// Check angle equality
    aeq,
    /// Multiply angle by a scalar
    amul,
    /// Divide by scalar with rounding
    adiv,
}

impl MakeOpDef for AngleOp {
    fn from_def(
        op_def: &hugr::extension::OpDef,
    ) -> Result<Self, hugr::extension::simple_op::OpLoadError>
    where
        Self: Sized,
    {
        hugr::extension::simple_op::try_from_name(op_def.name(), &ANGLE_EXTENSION_ID)
    }

    fn signature(&self) -> hugr::extension::SignatureFunc {
        match self {
            AngleOp::atrunc => {
                Signature::new(type_row![ANGLE_TYPE, USIZE_T], type_row![ANGLE_TYPE])
            }
            AngleOp::aadd | AngleOp::asub => {
                Signature::new(type_row![ANGLE_TYPE, ANGLE_TYPE], type_row![ANGLE_TYPE])
            }
            AngleOp::aneg => Signature::new_endo(type_row![ANGLE_TYPE]),
            AngleOp::anew => Signature::new(
                type_row![USIZE_T, USIZE_T],
                vec![sum_with_error(ANGLE_TYPE).into()],
            ),
            AngleOp::aparts => Signature::new(type_row![ANGLE_TYPE], type_row![USIZE_T, USIZE_T]),
            AngleOp::afromrad => {
                Signature::new(type_row![USIZE_T, FLOAT64_TYPE], type_row![ANGLE_TYPE])
            }
            AngleOp::atorad => Signature::new(type_row![ANGLE_TYPE], type_row![FLOAT64_TYPE]),
            AngleOp::aeq => Signature::new(type_row![ANGLE_TYPE, ANGLE_TYPE], type_row![BOOL_T]),
            AngleOp::amul | AngleOp::adiv => {
                Signature::new(type_row![ANGLE_TYPE, USIZE_T], type_row![ANGLE_TYPE])
            }
        }
        .into()
    }

    fn description(&self) -> String {
        match self {
            AngleOp::atrunc => "truncate an angle to one with a lower log-denominator with the nearest value, rounding down in [0, 2π) if necessary",
            AngleOp::aadd => "addition of angles",
            AngleOp::asub => "subtraction of the second angle from the first",
            AngleOp::aneg => "negation of an angle",
            AngleOp::anew => "construct angle from numerator and log-denominator, returning an error if invalid",
            AngleOp::aparts => "decompose angle into numerator and log-denominator",
            AngleOp::afromrad => "construct angle from radians, rounding given a log-denominator",
            AngleOp::atorad => "convert angle to radians",
            AngleOp::aeq => "check angle equality",
            AngleOp::amul => "multiply angle by a scalar",
            AngleOp::adiv => "Divide angle by an integer. If the integer is not a power of 2, or if the resulting denominator would exceed 2^64, the result is rounded to the nearest multiple of 2 pi / 2^ 64",
        }.to_owned()
    }

    fn extension(&self) -> hugr::extension::ExtensionId {
        ANGLE_EXTENSION_ID
    }

    // TODO constant folding
    // https://github.com/CQCL/tket2/issues/405
}

impl MakeRegisteredOp for AngleOp {
    fn extension_id(&self) -> hugr::extension::ExtensionId {
        ANGLE_EXTENSION_ID
    }

    fn registry<'s, 'r: 's>(&'s self) -> &'r hugr::extension::ExtensionRegistry {
        &super::REGISTRY
    }
}

pub(super) fn add_to_extension(extension: &mut Extension) {
    extension
        .add_type(
            ANGLE_TYPE_ID,
            vec![],
            "angle type expressed as dyadic rational multiples of 2π".to_owned(),
            TypeBound::Copyable.into(),
        )
        .unwrap();

    AngleOp::load_all_ops(extension).expect("add fail");
}

#[cfg(test)]
mod test {
    use hugr::ops::OpType;
    use strum::IntoEnumIterator;

    use super::*;

    #[test]
    fn test_angle_consts() {
        let const_a32_7 = ConstAngle::new(5, 7).unwrap();
        let const_a33_7 = ConstAngle::new(6, 7).unwrap();
        let const_a32_8 = ConstAngle::new(6, 8).unwrap();
        assert_ne!(const_a32_7, const_a33_7);
        assert_ne!(const_a32_7, const_a32_8);
        assert_eq!(const_a32_7, ConstAngle::new(5, 7).unwrap());

        assert_eq!(const_a32_7.get_type(), ANGLE_TYPE);
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

        assert!(ConstAngle::from_radians_rounding(54, 0.21874 * TAU).is_err());

        assert!(const_a32_7.equal_consts(&ConstAngle::new(5, 7).unwrap()));
        assert_ne!(const_a32_7, const_a33_7);

        assert_eq!(const_a32_8.name(), "a(2π*8/2^6)");
    }

    #[test]
    fn test_ops() {
        let ops = AngleOp::iter().collect::<Vec<_>>();
        for op in ops {
            let optype: OpType = op.into();
            assert_eq!(optype.cast(), Some(op));
        }
    }
}
