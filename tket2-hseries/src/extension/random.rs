//! This module defines the "tket2.qsystem.random" extension, which includes
//! random number generation (RNG) functions available for Quantinuum systems.

use std::sync::{Arc, Weak};

use derive_more::derive::Display;
use hugr::{
    builder::{BuildError, Dataflow},
    extension::{
        prelude::{option_type, UnwrapBuilder, PRELUDE_ID},
        simple_op::{try_from_name, MakeOpDef, MakeRegisteredOp},
        ExtensionBuildError, ExtensionId, ExtensionRegistry, ExtensionSet, OpDef, SignatureFunc,
        TypeDefBound, Version, PRELUDE,
    },
    std_extensions::arithmetic::{float_types::float64_type, int_types::int_type},
    types::{type_param::TypeParam, CustomType, Signature, Type, TypeBound},
    Extension, Wire,
};
use lazy_static::lazy_static;
use smol_str::SmolStr;
use strum::{EnumIter, EnumString, IntoStaticStr};

/// The extension ID for the RNG extension.
pub const EXTENSION_ID: ExtensionId = ExtensionId::new_unchecked("tket2.qsystem.random");
/// The version of the "tket2.qsystem.random" extension.
pub const EXTENSION_VERSION: Version = Version::new(0, 1, 0);

lazy_static! {
    /// The "tket2.qsystem.random" extension.
    pub static ref EXTENSION: Arc<Extension> = {
         Extension::new_arc(EXTENSION_ID, EXTENSION_VERSION, |ext, ext_ref| {
            ext.add_requirements(ExtensionSet::from_iter([PRELUDE_ID]));
            add_random_type_defs(ext, ext_ref).unwrap();
            RandomOp::load_all_ops( ext, ext_ref).unwrap();
        })
    };

    /// Extension registry including the "tket2.qsystem.random" extension and
    /// dependencies.
    pub static ref REGISTRY: ExtensionRegistry = ExtensionRegistry::new([
        EXTENSION.to_owned(),
        PRELUDE.to_owned(),
    ]);

    /// The name of the `tket2.qsystem.random.context` type.
    pub static ref CONTEXT_TYPE_NAME: SmolStr = SmolStr::new_inline("context");

    /// The type parameter for the seed of the RNG.
    pub static ref SEED: TypeParam =
    TypeParam::Type { b: TypeBound::Any };
}

fn add_random_type_defs(
    extension: &mut Extension,
    extension_ref: &Weak<Extension>,
) -> Result<(), ExtensionBuildError> {
    extension.add_type(
        CONTEXT_TYPE_NAME.to_owned(),
        vec![SEED.to_owned()],
        "The linear RNG context type".into(),
        TypeDefBound::any(),
        extension_ref,
    )?;
    Ok(())
}

/// An enum for RNG extension types.
pub enum RandomType {
    /// The linear RNG context type.
    RNGContext,
}

impl RandomType {
    fn get_type(&self, extension_ref: &Weak<Extension>) -> Type {
        match self {
            Self::RNGContext { .. } => CustomType::new(
                CONTEXT_TYPE_NAME.to_owned(),
                vec![int_type(6).into()],
                EXTENSION_ID,
                TypeBound::Any,
                extension_ref,
            ),
        }
        .into()
    }
}

#[derive(
    Clone,
    Copy,
    Debug,
    serde::Serialize,
    serde::Deserialize,
    Hash,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    EnumIter,
    IntoStaticStr,
    EnumString,
    Display,
)]
/// The operations provided by the random extension.
pub enum RandomOp {
    /// `fn random_int(RNGContext) -> (RNGContext, u32)`
    RandomInt,
    /// `fn random_float(RNGContext) -> (RNGContext, f32)`
    RandomFloat,
    /// `fn random_int_bounded(RNGContext, bound: u32) -> (RNGContext, u32)`
    RandomIntBounded,
    /// `fn new_rng_context(seed: u64) -> Option<RNGContext>` // return None on second call
    NewRNGContext,
    /// `fn delete_rng_context(RNGContext) -> ()`
    DeleteRNGContext,
}

impl MakeOpDef for RandomOp {
    fn init_signature(&self, extension_ref: &std::sync::Weak<Extension>) -> SignatureFunc {
        match self {
            RandomOp::RandomInt => Signature::new(
                vec![RandomType::RNGContext.get_type(extension_ref)],
                vec![RandomType::RNGContext.get_type(extension_ref), int_type(5)],
            ),
            RandomOp::RandomFloat => Signature::new(
                vec![RandomType::RNGContext.get_type(extension_ref)],
                vec![
                    RandomType::RNGContext.get_type(extension_ref),
                    float64_type(),
                ],
            ),
            RandomOp::RandomIntBounded => Signature::new(
                vec![RandomType::RNGContext.get_type(extension_ref), int_type(5)],
                vec![RandomType::RNGContext.get_type(extension_ref), int_type(5)],
            ),
            RandomOp::NewRNGContext => Signature::new(
                vec![int_type(6)],
                Type::from(option_type(RandomType::RNGContext.get_type(extension_ref))),
            ),
            RandomOp::DeleteRNGContext => {
                Signature::new(vec![RandomType::RNGContext.get_type(extension_ref)], vec![])
            }
        }
        .into()
    }

    fn from_def(op_def: &OpDef) -> Result<Self, hugr::extension::simple_op::OpLoadError> {
        try_from_name(op_def.name(), op_def.extension_id())
    }

    fn extension(&self) -> ExtensionId {
        EXTENSION_ID
    }

    fn extension_ref(&self) -> std::sync::Weak<Extension> {
        Arc::downgrade(&EXTENSION)
    }

    fn description(&self) -> String {
        match self {
            RandomOp::RandomInt => "Generate a random 32-bit unsigned integer.",
            RandomOp::RandomFloat => "Generate a random floating point value in the range [0,1).",
            RandomOp::RandomIntBounded => "Generate a random 32-bit unsigned integer less than `bound`.",
            RandomOp::NewRNGContext => {
                "Seed the RNG and return a new RNG context. Required before using other RNG ops, can be called only once."
            }
            RandomOp::DeleteRNGContext => "Discard the given RNG context.",
        }
        .to_string()
    }
}

impl MakeRegisteredOp for RandomOp {
    fn extension_id(&self) -> ExtensionId {
        EXTENSION_ID
    }

    fn extension_ref(&self) -> Weak<Extension> {
        Arc::downgrade(&EXTENSION)
    }
}

/// An extension trait for [Dataflow] providing methods to add
/// "tket2.qsystem.random" operations.
pub trait RandomOpBuilder: Dataflow + UnwrapBuilder {
    /// Add a "tket2.qsystem.random.random_int" op.
    fn add_random_int(&mut self, ctx: Wire) -> Result<[Wire; 2], BuildError> {
        Ok(self
            .add_dataflow_op(RandomOp::RandomInt, [ctx])?
            .outputs_arr())
    }

    /// Add a "tket2.qsystem.random.random_float" op.
    fn add_random_float(&mut self, ctx: Wire) -> Result<[Wire; 2], BuildError> {
        Ok(self
            .add_dataflow_op(RandomOp::RandomFloat, [ctx])?
            .outputs_arr())
    }

    /// Add a "tket2.qsystem.random.random_int_bounded" op.
    fn add_random_int_bounded(&mut self, ctx: Wire, bound: Wire) -> Result<[Wire; 2], BuildError> {
        Ok(self
            .add_dataflow_op(RandomOp::RandomIntBounded, [ctx, bound])?
            .outputs_arr())
    }

    /// Add a "tket2.qsystem.random.new_rng_context" op.
    fn add_new_rng_context(&mut self, seed: Wire) -> Result<Wire, BuildError> {
        Ok(self
            .add_dataflow_op(RandomOp::NewRNGContext, [seed])?
            .out_wire(0))
    }

    /// Add a "tket2.qsystem.random.delete_rng_context" op.
    fn add_delete_rng_context(&mut self, ctx: Wire) -> Result<(), BuildError> {
        self.add_dataflow_op(RandomOp::DeleteRNGContext, [ctx])?;
        Ok(())
    }
}

impl<D: Dataflow> RandomOpBuilder for D {}

#[cfg(test)]
mod test {
    use hugr::ops::{NamedOp, Value};
    use hugr::std_extensions::arithmetic::int_types::ConstInt;

    use hugr::builder::{DataflowHugr, FunctionBuilder};
    use strum::IntoEnumIterator;

    use super::*;

    #[test]
    fn create_extension() {
        assert_eq!(EXTENSION.name(), &EXTENSION_ID);

        for o in RandomOp::iter() {
            assert_eq!(
                RandomOp::from_def(EXTENSION.get_op(&o.name()).unwrap()),
                Ok(o)
            );
        }
    }

    #[test]
    fn test_random_op_builder() {
        let hugr = {
            let mut func_builder = FunctionBuilder::new(
                "random_op_builder",
                Signature::new(vec![], vec![int_type(5)]),
            )
            .unwrap();

            let seed =
                func_builder.add_load_const(Value::from(ConstInt::new_u(6, 123456).unwrap()));
            let maybe_ctx = func_builder.add_new_rng_context(seed).unwrap();
            let [ctx] = func_builder
                .build_unwrap_sum(
                    1,
                    option_type(RandomType::RNGContext.get_type(&Arc::downgrade(&EXTENSION))),
                    maybe_ctx,
                )
                .unwrap();
            let bound = func_builder.add_load_const(Value::from(ConstInt::new_u(5, 100).unwrap()));
            let [ctx, _] = func_builder.add_random_int_bounded(ctx, bound).unwrap();
            let [ctx, _] = func_builder.add_random_float(ctx).unwrap();
            let [ctx, rnd] = func_builder.add_random_int(ctx).unwrap();
            func_builder.add_delete_rng_context(ctx).unwrap();
            func_builder.finish_hugr_with_outputs([rnd]).unwrap()
        };
        hugr.validate().unwrap()
    }
}
