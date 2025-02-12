//! This module contains utility functions for Quantinuum systems.

use std::sync::Arc;

use derive_more::derive::Display;
use hugr::{
    extension::{
        prelude::usize_t,
        simple_op::{try_from_name, MakeOpDef},
        ExtensionId, ExtensionRegistry, ExtensionSet, OpDef, SignatureFunc, Version, PRELUDE,
    },
    type_row,
    types::Signature,
    Extension,
};
use lazy_static::lazy_static;
use strum::{EnumIter, EnumString, IntoStaticStr};

/// The extension ID for the utils extension.
pub const EXTENSION_ID: ExtensionId = ExtensionId::new_unchecked("tket2.qsystem.utils");
/// The version of the "tket2.qsystem.utils" extension.
pub const EXTENSION_VERSION: Version = Version::new(0, 1, 0);

lazy_static! {
    /// The "tket2.qsystem" extension.
    pub static ref EXTENSION: Arc<Extension> = {
         Extension::new_arc(EXTENSION_ID, EXTENSION_VERSION, |ext, ext_ref| {
            ext.add_requirements(ExtensionSet::from_iter([
                PRELUDE.name(),
            ].into_iter().cloned()));
            Utils::load_all_ops( ext, ext_ref).unwrap();
        })
    };

    /// Extension registry including the "tket2.qsystem" extension and
    /// dependencies.
    pub static ref REGISTRY: ExtensionRegistry = ExtensionRegistry::new([
        EXTENSION.to_owned(),
        PRELUDE.to_owned(),
    ]);
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
/// The operations provided by the utils extension.
pub enum Utils {
    /// fn get_current_shot() -> usize
    GetCurrentShot,
}

impl MakeOpDef for Utils {
    fn init_signature(&self, _extension_ref: &std::sync::Weak<Extension>) -> SignatureFunc {
        match self {
            Utils::GetCurrentShot => Signature::new(type_row![], usize_t()),
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
            Utils::GetCurrentShot => "Get current shot number.",
        }
        .to_string()
    }
}
