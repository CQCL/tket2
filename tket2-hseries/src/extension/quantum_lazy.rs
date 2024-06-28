//! TODO docs
use hugr::{
    builder::{BuildError, Dataflow},
    extension::{
        prelude::{BOOL_T, QB_T},
        simple_op::{try_from_name, MakeExtensionOp, MakeOpDef, MakeRegisteredOp},
        ExtensionId, ExtensionRegistry, OpDef, SignatureFunc, PRELUDE,
    },
    ops::{CustomOp, NamedOp as _, OpType},
    types::FunctionType,
    Extension, Wire,
};

use lazy_static::lazy_static;
use strum_macros::{EnumIter, EnumString, IntoStaticStr};

use crate::extension::lazy;

use super::lazy::lazy_type;

/// TODO docs
pub const EXTENSION_ID: ExtensionId = ExtensionId::new_unchecked("tket2.quantum.lazy");

lazy_static! {
    /// The "tket2.quantum.lazy" extension
    pub static ref EXTENSION: Extension = {
        let mut ext = Extension::new(EXTENSION_ID);
        LazyQuantumOp::load_all_ops(&mut ext).unwrap();
        ext
    };

    /// TODO docs
    pub static ref REGISTRY: ExtensionRegistry = ExtensionRegistry::try_new([
        lazy::EXTENSION.to_owned(),
        PRELUDE.to_owned(),
        EXTENSION.to_owned()
    ]).unwrap();
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
)]
#[allow(missing_docs)]
#[non_exhaustive]
pub enum LazyQuantumOp {
    LazyMeasure,
}

impl MakeOpDef for LazyQuantumOp {
    fn signature(&self) -> SignatureFunc {
        FunctionType::new(QB_T, vec![QB_T, lazy_type(BOOL_T)]).into()
    }

    fn from_def(op_def: &OpDef) -> Result<Self, hugr::extension::simple_op::OpLoadError> {
        try_from_name(op_def.name())
    }
}

impl MakeRegisteredOp for LazyQuantumOp {
    fn extension_id(&self) -> ExtensionId {
        EXTENSION_ID
    }

    fn registry<'s, 'r: 's>(&'s self) -> &'r ExtensionRegistry {
        &REGISTRY
    }
}

impl TryFrom<&OpType> for LazyQuantumOp {
    type Error = ();
    fn try_from(value: &OpType) -> Result<Self, Self::Error> {
        let Some(custom_op) = value.as_custom_op() else {
            Err(())?
        };
        match custom_op {
            CustomOp::Extension(ext) => Self::from_extension_op(ext).ok(),
            CustomOp::Opaque(opaque) if opaque.extension() == &EXTENSION_ID => {
                try_from_name(opaque.name()).ok()
            }
            _ => None,
        }
        .ok_or(())
    }
}

/// TODO docs
pub trait LazyQuantumOpBuilder: Dataflow {
    /// TODO docs
    fn add_lazy_measure(&mut self, qb: Wire) -> Result<[Wire; 2], BuildError> {
        Ok(self
            .add_dataflow_op(LazyQuantumOp::LazyMeasure, [qb])?
            .outputs_arr())
    }
}

impl<D: Dataflow> LazyQuantumOpBuilder for D {}

#[cfg(test)]
mod test {
    use std::sync::Arc;

    use cool_asserts::assert_matches;
    use hugr::{
        builder::{DataflowHugr, FunctionBuilder},
        ops::NamedOp,
    };
    use lazy::LazyOpBuilder as _;
    use strum::IntoEnumIterator as _;

    use super::*;

    fn get_opdef(op: impl NamedOp) -> Option<&'static Arc<OpDef>> {
        EXTENSION.get_op(&op.name())
    }

    #[test]
    fn create_extension() {
        assert_eq!(EXTENSION.name(), &EXTENSION_ID);

        for o in LazyQuantumOp::iter() {
            assert_eq!(LazyQuantumOp::from_def(get_opdef(o).unwrap()), Ok(o));
        }
    }

    #[test]
    fn circuit() {
        let hugr = {
            let mut func_builder = FunctionBuilder::new(
                "circuit",
                FunctionType::new(QB_T, vec![QB_T, BOOL_T]).into(),
            )
            .unwrap();
            let [qb] = func_builder.input_wires_arr();
            let [qb, lazy_b] = func_builder.add_lazy_measure(qb).unwrap();
            let [b] = func_builder.add_read(lazy_b, BOOL_T).unwrap();
            func_builder
                .finish_hugr_with_outputs([qb, b], &REGISTRY)
                .unwrap()
        };
        assert_matches!(hugr.validate(&REGISTRY), Ok(_));
    }
}
