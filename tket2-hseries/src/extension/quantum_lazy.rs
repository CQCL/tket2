//! This module defines the Hugr extension used to represent Lazy Quantum
//! Operations.
//!
//! Laziness is represented by returning `tket2.futures.Future` classical
//! values. Qubits are never lazy.
use hugr::{
    builder::{BuildError, Dataflow},
    extension::{
        prelude::{BOOL_T, QB_T},
        simple_op::{try_from_name, MakeOpDef, MakeRegisteredOp, OpLoadError},
        ExtensionId, ExtensionRegistry, OpDef, SignatureFunc, PRELUDE,
    },
    ops::{NamedOp as _, OpType},
    types::Signature,
    Extension, Wire,
};

use lazy_static::lazy_static;
use strum_macros::{EnumIter, EnumString, IntoStaticStr};

use crate::extension::futures;

use super::futures::future_type;

/// The "tket2.quantum.lazy" extension id.
pub const EXTENSION_ID: ExtensionId = ExtensionId::new_unchecked("tket2.quantum.lazy");

lazy_static! {
    /// The "tket2.quantum.lazy" extension.
    pub static ref EXTENSION: Extension = {
        let mut ext = Extension::new(EXTENSION_ID);
        LazyQuantumOp::load_all_ops(&mut ext).unwrap();
        ext
    };

    /// Extension registry including the "tket2.quantum.lazy" extension and
    /// dependencies.
    pub static ref REGISTRY: ExtensionRegistry = ExtensionRegistry::try_new([
        futures::EXTENSION.to_owned(),
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
    Measure,
}

impl MakeOpDef for LazyQuantumOp {
    fn signature(&self) -> SignatureFunc {
        match self {
            Self::Measure => Signature::new(QB_T, vec![QB_T, future_type(BOOL_T)]).into(),
        }
    }

    fn from_def(op_def: &OpDef) -> Result<Self, hugr::extension::simple_op::OpLoadError> {
        try_from_name(op_def.name(), &EXTENSION_ID)
    }

    fn extension(&self) -> ExtensionId {
        EXTENSION_ID
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
    type Error = OpLoadError;
    fn try_from(value: &OpType) -> Result<Self, Self::Error> {
        Self::from_op(
            value
                .as_custom_op()
                .ok_or(OpLoadError::NotMember(value.name().into()))?,
        )
    }
}

/// An extension trait for [Dataflow] providing methods to add
/// "tket2.quantum.lazy" operations.
pub trait LazyQuantumOpBuilder: Dataflow {
    /// Add a "tket2.quantum.lazy.Measure" op.
    fn add_lazy_measure(&mut self, qb: Wire) -> Result<[Wire; 2], BuildError> {
        Ok(self
            .add_dataflow_op(LazyQuantumOp::Measure, [qb])?
            .outputs_arr())
    }
}

impl<D: Dataflow> LazyQuantumOpBuilder for D {}

#[cfg(test)]
mod test {
    use std::sync::Arc;

    use cool_asserts::assert_matches;
    use futures::FutureOpBuilder as _;
    use hugr::{
        builder::{DataflowHugr, FunctionBuilder},
        ops::NamedOp,
    };
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
            let mut func_builder =
                FunctionBuilder::new("circuit", Signature::new(QB_T, vec![QB_T, BOOL_T])).unwrap();
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
