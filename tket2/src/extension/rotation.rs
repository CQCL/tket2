use hugr::builder::{BuildError, Dataflow};
use hugr::extension::prelude::{sum_with_error, BOOL_T, USIZE_T};
use hugr::extension::simple_op::{MakeOpDef, MakeRegisteredOp};
use hugr::extension::{ExtensionId, ExtensionSet, Version};
use hugr::ops::constant::{downcast_equal_consts, CustomConst};
use hugr::std_extensions::arithmetic::float_types::FLOAT64_TYPE;
use hugr::{type_row, Wire};
use hugr::{
    types::{ConstTypeError, CustomType, Signature, Type, TypeBound},
    Extension,
};
use smol_str::SmolStr;
use std::f64::consts::TAU;
use strum::{EnumIter, EnumString, IntoStaticStr};

use lazy_static::lazy_static;

/// Name of tket 2 rotation extension.
pub const ROTATION_EXTENSION_ID: ExtensionId = ExtensionId::new_unchecked("tket2.rotation");

/// Current version of the TKET 2 rotation extension
pub const ROTATION_EXTENSION_VERSION: Version = Version::new(0, 1, 0);

lazy_static! {
    /// The extension definition for TKET2 rotation type and ops.
    pub static ref ROTATION_EXTENSION: Extension = {
        let mut e = Extension::new(ROTATION_EXTENSION_ID, ROTATION_EXTENSION_VERSION);
        add_to_extension(&mut e);
        e
    };
}

/// Identifier for the rotation type.
const ROTATION_TYPE_ID: SmolStr = SmolStr::new_inline("rotation");
/// Dyadic rational rotation type (as [CustomType])
pub const ROTATION_CUSTOM_TYPE: CustomType =
    CustomType::new_simple(ROTATION_TYPE_ID, ROTATION_EXTENSION_ID, TypeBound::Copyable);

/// Type representing a rotation that is a dyadic rational multiple of 2π (as [Type])
pub const ROTATION_TYPE: Type = Type::new_extension(ROTATION_CUSTOM_TYPE);

/// The largest permitted log-denominator.
pub const LOG_DENOM_MAX: u8 = 53;

const fn is_valid_log_denom(n: u8) -> bool {
    n <= LOG_DENOM_MAX
}

/// A rotation
#[derive(Clone, Debug, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ConstRotation {
    log_denom: u8,
    value: u64,
}

impl ConstRotation {
    /// The constant π
    pub const PI: Self = Self::new_unchecked(1, 1);
    /// The constant 2π
    pub const TAU: Self = Self::new_unchecked(0, 0);
    /// The constant π/2
    pub const PI_2: Self = Self::new_unchecked(2, 1);
    /// The constant π/4
    pub const PI_4: Self = Self::new_unchecked(3, 1);

    /// Create a new [`ConstRotation`] from a log-denominator and a numerator without
    /// checking for validity.
    const fn new_unchecked(log_denom: u8, value: u64) -> Self {
        Self { log_denom, value }
    }
    /// Create a new [`ConstRotation`] from a log-denominator and a numerator
    pub fn new(log_denom: u8, value: u64) -> Result<Self, ConstTypeError> {
        if !is_valid_log_denom(log_denom) {
            return Err(ConstTypeError::CustomCheckFail(
                hugr::types::CustomCheckFailure::Message(
                    "Invalid rotation log-denominator.".to_owned(),
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

    /// Create a new [`ConstRotation`] from a log-denominator and a floating-point value in radians,
    /// rounding to the nearest corresponding value. (Ties round away from zero.)
    pub fn from_radians_rounding(log_denom: u8, theta: f64) -> Result<Self, ConstTypeError> {
        if !is_valid_log_denom(log_denom) {
            return Err(ConstTypeError::CustomCheckFail(
                hugr::types::CustomCheckFailure::Message(
                    "Invalid rotation log-denominator.".to_owned(),
                ),
            ));
        }
        let a = (((1u64 << log_denom) as f64) * theta / TAU).round() as i64;
        Ok(Self {
            log_denom,
            value: a.rem_euclid(1i64 << log_denom) as u64,
        })
    }

    /// Create a new [`ConstRotation`] from a floating-point value in radians,
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
impl CustomConst for ConstRotation {
    fn name(&self) -> SmolStr {
        format!("a(2π*{}/2^{})", self.value, self.log_denom).into()
    }

    fn get_type(&self) -> Type {
        ROTATION_TYPE
    }

    fn equal_consts(&self, other: &dyn CustomConst) -> bool {
        downcast_equal_consts(self, other)
    }
    fn extension_reqs(&self) -> ExtensionSet {
        ExtensionSet::singleton(&ROTATION_EXTENSION_ID)
    }
}

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq, EnumIter, IntoStaticStr, EnumString)]
#[allow(missing_docs, non_camel_case_types)]
#[non_exhaustive]
/// Rotation operations
pub enum RotationOp {
    /// Truncate a rotation to one with a lower log-denominator with the nearest value, rounding down in [0, 2π) if necessary
    atrunc,
    /// Addition of rotations
    aadd,
    /// Subtraction of the second rotation from the first
    asub,
    /// Negation of a rotation
    aneg,
    /// Construct rotation from numerator and log-denominator
    anew,
    /// Decompose rotation into numerator and log-denominator
    aparts,
    /// Construct rotation from radians
    afromrad,
    /// Convert rotation to radians
    atorad,
    /// Check rotation equality
    aeq,
    /// Multiply rotation by a scalar
    amul,
    /// Divide by scalar with rounding
    adiv,
}

impl MakeOpDef for RotationOp {
    fn from_def(
        op_def: &hugr::extension::OpDef,
    ) -> Result<Self, hugr::extension::simple_op::OpLoadError>
    where
        Self: Sized,
    {
        hugr::extension::simple_op::try_from_name(op_def.name(), op_def.extension())
    }

    fn signature(&self) -> hugr::extension::SignatureFunc {
        match self {
            RotationOp::atrunc => {
                Signature::new(type_row![ROTATION_TYPE, USIZE_T], type_row![ROTATION_TYPE])
            }
            RotationOp::aadd | RotationOp::asub => Signature::new(
                type_row![ROTATION_TYPE, ROTATION_TYPE],
                type_row![ROTATION_TYPE],
            ),
            RotationOp::aneg => Signature::new_endo(type_row![ROTATION_TYPE]),
            RotationOp::anew => Signature::new(
                type_row![USIZE_T, USIZE_T],
                vec![sum_with_error(ROTATION_TYPE).into()],
            ),
            RotationOp::aparts => {
                Signature::new(type_row![ROTATION_TYPE], type_row![USIZE_T, USIZE_T])
            }
            RotationOp::afromrad => {
                Signature::new(type_row![USIZE_T, FLOAT64_TYPE], type_row![ROTATION_TYPE])
            }
            RotationOp::atorad => Signature::new(type_row![ROTATION_TYPE], type_row![FLOAT64_TYPE]),
            RotationOp::aeq => {
                Signature::new(type_row![ROTATION_TYPE, ROTATION_TYPE], type_row![BOOL_T])
            }
            RotationOp::amul | RotationOp::adiv => {
                Signature::new(type_row![ROTATION_TYPE, USIZE_T], type_row![ROTATION_TYPE])
            }
        }
        .into()
    }

    fn description(&self) -> String {
        match self {
            RotationOp::atrunc => "truncate a rotation to one with a lower log-denominator with the nearest value, rounding down in [0, 2π) if necessary",
            RotationOp::aadd => "addition of rotations",
            RotationOp::asub => "subtraction of the second rotation from the first",
            RotationOp::aneg => "negation of a rotation",
            RotationOp::anew => "construct rotation from numerator and log-denominator, returning an error if invalid",
            RotationOp::aparts => "decompose rotation into numerator and log-denominator",
            RotationOp::afromrad => "construct rotation from radians, rounding given a log-denominator",
            RotationOp::atorad => "convert rotation to radians",
            RotationOp::aeq => "check rotation equality",
            RotationOp::amul => "multiply rotation by a scalar",
            RotationOp::adiv => "Divide rotation by an integer. If the integer is not a power of 2, or if the resulting denominator would exceed 2^64, the result is rounded to the nearest multiple of 2 pi / 2^ 64",
        }.to_owned()
    }

    fn extension(&self) -> hugr::extension::ExtensionId {
        ROTATION_EXTENSION_ID
    }

    // TODO constant folding
    // https://github.com/CQCL/tket2/issues/405
}

impl MakeRegisteredOp for RotationOp {
    fn extension_id(&self) -> hugr::extension::ExtensionId {
        ROTATION_EXTENSION_ID
    }

    fn registry<'s, 'r: 's>(&'s self) -> &'r hugr::extension::ExtensionRegistry {
        &super::REGISTRY
    }
}

pub(super) fn add_to_extension(extension: &mut Extension) {
    extension
        .add_type(
            ROTATION_TYPE_ID,
            vec![],
            "rotation type expressed as dyadic rational multiples of 2π".to_owned(),
            TypeBound::Copyable.into(),
        )
        .unwrap();

    RotationOp::load_all_ops(extension).expect("add fail");
}

/// An extension trait for [Dataflow] providing methods to add
/// "tket2.rotation" operations.
pub trait RotationOpBuilder: Dataflow {
    /// Add a "tket2.rotation.atrunc" op.
    fn add_atrunc(&mut self, rotation: Wire, log_denom: Wire) -> Result<Wire, BuildError> {
        Ok(self
            .add_dataflow_op(RotationOp::atrunc, [rotation, log_denom])?
            .out_wire(0))
    }
    /// Add a "tket2.rotation.aadd" op.
    fn add_aadd(&mut self, rotation1: Wire, rotation2: Wire) -> Result<Wire, BuildError> {
        Ok(self
            .add_dataflow_op(RotationOp::aadd, [rotation1, rotation2])?
            .out_wire(0))
    }

    /// Add a "tket2.rotation.asub" op.
    fn add_asub(&mut self, rotation1: Wire, rotation2: Wire) -> Result<Wire, BuildError> {
        Ok(self
            .add_dataflow_op(RotationOp::asub, [rotation1, rotation2])?
            .out_wire(0))
    }

    /// Add a "tket2.rotation.aneg" op.
    fn add_aneg(&mut self, rotation: Wire) -> Result<Wire, BuildError> {
        Ok(self
            .add_dataflow_op(RotationOp::aneg, [rotation])?
            .out_wire(0))
    }

    /// Add a "tket2.rotation.anew" op.
    fn add_anew(&mut self, numerator: Wire, log_denominator: Wire) -> Result<Wire, BuildError> {
        Ok(self
            .add_dataflow_op(RotationOp::anew, [numerator, log_denominator])?
            .out_wire(0))
    }

    /// Add a "tket2.rotation.aparts" op.
    fn add_aparts(&mut self, rotation: Wire) -> Result<[Wire; 2], BuildError> {
        Ok(self
            .add_dataflow_op(RotationOp::aparts, [rotation])?
            .outputs_arr())
    }

    /// Add a "tket2.rotation.afromrad" op.
    fn add_afromrad(&mut self, log_denominator: Wire, radians: Wire) -> Result<Wire, BuildError> {
        Ok(self
            .add_dataflow_op(RotationOp::afromrad, [log_denominator, radians])?
            .out_wire(0))
    }

    /// Add a "tket2.rotation.atorad" op.
    fn add_atorad(&mut self, rotation: Wire) -> Result<Wire, BuildError> {
        Ok(self
            .add_dataflow_op(RotationOp::atorad, [rotation])?
            .out_wire(0))
    }

    /// Add a "tket2.rotation.aeq" op.
    fn add_aeq(&mut self, rotation1: Wire, rotation2: Wire) -> Result<Wire, BuildError> {
        Ok(self
            .add_dataflow_op(RotationOp::aeq, [rotation1, rotation2])?
            .out_wire(0))
    }

    /// Add a "tket2.rotation.amul" op.
    fn add_amul(&mut self, rotation: Wire, scalar: Wire) -> Result<Wire, BuildError> {
        Ok(self
            .add_dataflow_op(RotationOp::amul, [rotation, scalar])?
            .out_wire(0))
    }

    /// Add a "tket2.rotation.adiv" op.
    fn add_adiv(&mut self, rotation: Wire, scalar: Wire) -> Result<Wire, BuildError> {
        Ok(self
            .add_dataflow_op(RotationOp::adiv, [rotation, scalar])?
            .out_wire(0))
    }
}

impl<D: Dataflow> RotationOpBuilder for D {}

#[cfg(test)]
mod test {
    use hugr::{
        builder::{DFGBuilder, DataflowHugr},
        ops::OpType,
    };
    use strum::IntoEnumIterator;

    use crate::extension::REGISTRY;

    use super::*;

    #[test]
    fn test_rotation_consts() {
        let const_a32_7 = ConstRotation::new(5, 7).unwrap();
        let const_a33_7 = ConstRotation::new(6, 7).unwrap();
        let const_a32_8 = ConstRotation::new(6, 8).unwrap();
        assert_ne!(const_a32_7, const_a33_7);
        assert_ne!(const_a32_7, const_a32_8);
        assert_eq!(const_a32_7, ConstRotation::new(5, 7).unwrap());

        assert_eq!(const_a32_7.get_type(), ROTATION_TYPE);
        assert!(matches!(
            ConstRotation::new(3, 256),
            Err(ConstTypeError::CustomCheckFail(_))
        ));
        assert!(matches!(
            ConstRotation::new(54, 256),
            Err(ConstTypeError::CustomCheckFail(_))
        ));
        let const_af1 = ConstRotation::from_radians_rounding(5, 0.21874 * TAU).unwrap();
        assert_eq!(const_af1.value(), 7);
        assert_eq!(const_af1.log_denom(), 5);

        assert!(ConstRotation::from_radians_rounding(54, 0.21874 * TAU).is_err());

        assert!(const_a32_7.equal_consts(&ConstRotation::new(5, 7).unwrap()));
        assert_ne!(const_a32_7, const_a33_7);

        assert_eq!(const_a32_8.name(), "a(2π*8/2^6)");
    }

    #[test]
    fn test_ops() {
        let ops = RotationOp::iter().collect::<Vec<_>>();
        for op in ops {
            let optype: OpType = op.into();
            assert_eq!(optype.cast(), Some(op));
        }
    }

    #[test]
    fn test_builder() {
        let mut builder =
            DFGBuilder::new(Signature::new(vec![ROTATION_TYPE, USIZE_T], vec![BOOL_T])).unwrap();

        let [rotation, scalar] = builder.input_wires_arr();
        let radians = builder.add_atorad(rotation).unwrap();
        let rotation = builder.add_afromrad(scalar, radians).unwrap();
        let rotation = builder.add_amul(rotation, scalar).unwrap();
        let rotation = builder.add_adiv(rotation, scalar).unwrap();
        let rotation = builder.add_aadd(rotation, rotation).unwrap();
        let rotation = builder.add_asub(rotation, rotation).unwrap();
        let [num, log_denom] = builder.add_aparts(rotation).unwrap();
        let _rotation_sum = builder.add_anew(num, log_denom).unwrap();
        let rotation = builder.add_aneg(rotation).unwrap();
        let rotation = builder.add_atrunc(rotation, log_denom).unwrap();
        let bool = builder.add_aeq(rotation, rotation).unwrap();

        let _hugr = builder.finish_hugr_with_outputs([bool], &REGISTRY).unwrap();
    }

    #[rstest::rstest]
    fn const_rotation_statics(
        #[values(
            ConstRotation::TAU,
            ConstRotation::PI,
            ConstRotation::PI_2,
            ConstRotation::PI_4
        )]
        konst: ConstRotation,
    ) {
        assert_eq!(
            ConstRotation::new(konst.log_denom(), konst.value()),
            Ok(konst)
        );
    }
}
