//! This module defines the "tket2.qsystem.utils" extension, which includes the
//! utility functions available for Quantinuum systems.

use std::sync::{Arc, Weak};

use derive_more::derive::Display;
use hugr::{
    builder::{BuildError, Dataflow},
    extension::{
        prelude::{qb_t, UnwrapBuilder},
        simple_op::{try_from_name, MakeOpDef, MakeRegisteredOp},
        ExtensionId, ExtensionRegistry, ExtensionSet, OpDef, SignatureFunc, Version, PRELUDE,
    },
    std_extensions::{arithmetic::int_types::int_type, collections::array::array_type},
    type_row,
    types::Signature,
    Extension, Wire,
};
use lazy_static::lazy_static;
use strum::{EnumIter, EnumString, IntoStaticStr};

/// The extension ID for the utils extension.
pub const EXTENSION_ID: ExtensionId = ExtensionId::new_unchecked("tket2.qsystem.utils");
/// The version of the "tket2.qsystem.utils" extension.
pub const EXTENSION_VERSION: Version = Version::new(0, 1, 0);

lazy_static! {
    /// The "tket2.qsystem.utils" extension.
    pub static ref EXTENSION: Arc<Extension> = {
         Extension::new_arc(EXTENSION_ID, EXTENSION_VERSION, |ext, ext_ref| {
            ext.add_requirements(ExtensionSet::from_iter([
                PRELUDE.name(),
            ].into_iter().cloned()));
            UtilsOp::load_all_ops( ext, ext_ref).unwrap();
        })
    };

    /// Extension registry including the "tket2.qsystem.utils" extension and
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
pub enum UtilsOp {
    /// `fn get_current_shot() -> usize`
    GetCurrentShot,
    /// `fn order_in_zones(array_type(16, qb_t))`
    OrderInZones,
}

impl MakeOpDef for UtilsOp {
    fn init_signature(&self, _extension_ref: &std::sync::Weak<Extension>) -> SignatureFunc {
        match self {
            UtilsOp::GetCurrentShot => Signature::new(type_row![], int_type(6)),
            UtilsOp::OrderInZones => Signature::new(array_type(16, qb_t()), array_type(16, qb_t())),
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
            UtilsOp::GetCurrentShot => "Get current shot number.",
            UtilsOp::OrderInZones => "Order qubits in gating zones.",
        }
        .to_string()
    }
}

impl MakeRegisteredOp for UtilsOp {
    fn extension_id(&self) -> ExtensionId {
        EXTENSION_ID
    }

    fn extension_ref(&self) -> Weak<Extension> {
        Arc::downgrade(&EXTENSION)
    }
}

/// An extension trait for [Dataflow] providing methods to add
/// "tket2.qsystem.utils" operations.
pub trait UtilsOpBuilder: Dataflow + UnwrapBuilder {
    /// Add a "tket2.qsystem.utils.GetCurrentShot" op.
    fn add_get_current_shot(&mut self) -> Result<Wire, BuildError> {
        Ok(self
            .add_dataflow_op(UtilsOp::GetCurrentShot, [])?
            .out_wire(0))
    }

    /// Add a "tket2.qsystem.utils.OrderInZones" op.
    fn add_order_in_zones(&mut self, qubits: Wire) -> Result<Wire, BuildError> {
        Ok(self
            .add_dataflow_op(UtilsOp::OrderInZones, [qubits])?
            .out_wire(0))
    }
}

impl<D: Dataflow> UtilsOpBuilder for D {}

#[cfg(test)]
mod test {
    use std::sync::Arc;

    use hugr::builder::{DataflowHugr, FunctionBuilder};
    use hugr::ops::NamedOp;
    use strum::IntoEnumIterator;

    use super::*;

    fn get_opdef(op: impl NamedOp) -> Option<&'static Arc<OpDef>> {
        EXTENSION.get_op(&op.name())
    }

    #[test]
    fn create_extension() {
        assert_eq!(EXTENSION.name(), &EXTENSION_ID);

        for o in UtilsOp::iter() {
            assert_eq!(UtilsOp::from_def(get_opdef(o).unwrap()), Ok(o));
        }
    }

    #[test]
    fn get_current_shot() {
        let hugr = {
            let mut func_builder = FunctionBuilder::new(
                "get_current_shot",
                Signature::new(vec![], vec![int_type(6)]),
            )
            .unwrap();
            let shot = func_builder.add_get_current_shot().unwrap();
            func_builder.finish_hugr_with_outputs([shot]).unwrap()
        };
        hugr.validate().unwrap()
    }

    #[test]
    fn order_in_zones() {
        let hugr = {
            let mut func_builder = FunctionBuilder::new(
                "order_in_zones",
                Signature::new(vec![array_type(16, qb_t())], vec![array_type(16, qb_t())]),
            )
            .unwrap();
            let [qubits_in] = func_builder.input_wires_arr();
            let qubits_out = func_builder.add_order_in_zones(qubits_in).unwrap();
            func_builder.finish_hugr_with_outputs([qubits_out]).unwrap()
        };
        hugr.validate().unwrap()
    }
}
