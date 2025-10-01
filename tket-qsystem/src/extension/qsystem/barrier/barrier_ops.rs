//! Barrier-specific operations that use generic container unpacking/repacking.
//!
//! This module provides barrier-specific functionality that leverages the generic
//! container operations for unpacking and repacking container types, but focuses
//! specifically on qubit extraction and runtime barrier insertion.

use std::sync::{Arc, LazyLock};

use hugr::{
    builder::{BuildError, DFGBuilder, Dataflow, DataflowHugr},
    extension::{prelude::qb_t, Extension},
    ops::ExtensionOp,
    std_extensions::collections::array::array_type,
    types::{
        type_param::TypeParam, FuncValueType, PolyFuncTypeRV, Signature, TypeArg, TypeBound, TypeRV,
    },
    Hugr, Wire,
};

use crate::extension::qsystem::{cached_extensions::ExtensionCache, QSystemOpBuilder};

/// Temporary extension name for barrier-specific operations.
pub(super) const TEMP_BARRIER_EXT_NAME: hugr::hugr::IdentList =
    hugr::hugr::IdentList::new_static_unchecked("__tket.barrier.temp");

// Barrier-specific operation names.
pub(super) const WRAPPED_BARRIER_NAME: hugr::ops::OpName =
    hugr::ops::OpName::new_static("wrapped_barrier");

pub static TEMP_BARRIER_EXT: LazyLock<Arc<Extension>> = LazyLock::new(|| {
    Extension::new_arc(
        TEMP_BARRIER_EXT_NAME,
        hugr::extension::Version::new(0, 0, 0),
        |ext, ext_ref| {
            // version of runtime barrier that takes a variable number of qubits
            ext.add_op(
                WRAPPED_BARRIER_NAME,
                Default::default(),
                PolyFuncTypeRV::new(
                    vec![TypeParam::new_list_type(TypeBound::Linear)],
                    FuncValueType::new_endo(TypeRV::new_row_var_use(0, TypeBound::Linear)),
                ),
                ext_ref,
            )
            .unwrap();
        },
    )
});

/// Factory for creating barrier-specific operations that use generic container operations.
///
/// This factory focuses on barrier-specific functionality like runtime barrier insertion,
/// while delegating generic container unpacking/repacking to the ContainerOperationFactory.
pub struct BarrierOperationFactory {
    cache: ExtensionCache,
}

impl BarrierOperationFactory {
    /// Create a new instance of the BarrierOperationFactory.
    pub fn new() -> Self {
        Self {
            cache: ExtensionCache::new(),
        }
    }

    /// Build a runtime barrier across the given qubit wires using external cache
    pub fn build_runtime_barrier(
        &mut self,
        builder: &mut impl Dataflow,
        qubit_wires: Vec<Wire>,
    ) -> Result<hugr::builder::handle::Outputs, BuildError> {
        let size = qubit_wires.len();
        let qb_row = vec![qb_t(); size];
        let args = [TypeArg::List(
            qb_row.clone().into_iter().map(Into::into).collect(),
        )];
        let op = ExtensionOp::new(
            TEMP_BARRIER_EXT
                .get_op(&WRAPPED_BARRIER_NAME)
                .unwrap()
                .clone(),
            args.clone(),
        )
        .unwrap();
        let mangle_args: &[TypeArg] = &[TypeArg::BoundedNat(size as u64)];
        self.cache.cache_function(&op, mangle_args, |func_b| {
            func_b.build_wrapped_barrier(func_b.input_wires())
        })?;
        Ok(builder.add_dataflow_op(op, qubit_wires)?.outputs())
    }

    pub fn extension_cache(&self) -> &ExtensionCache {
        &self.cache
    }
    pub fn extension_cache_mut(&mut self) -> &mut ExtensionCache {
        &mut self.cache
    }
    pub fn into_function_map(self) -> impl Iterator<Item = (ExtensionOp, Hugr)> {
        self.cache.into_iter()
    }
}

impl Default for BarrierOperationFactory {
    fn default() -> Self {
        Self::new()
    }
}

/// Build a runtime barrier operation for an array of qubits
pub fn build_runtime_barrier_op(array_size: u64) -> Result<Hugr, BuildError> {
    let mut barr_builder = DFGBuilder::new(Signature::new_endo(array_type(array_size, qb_t())))?;
    let array_wire = barr_builder.input().out_wire(0);
    let out = barr_builder.add_runtime_barrier(array_wire, array_size)?;
    barr_builder.finish_hugr_with_outputs([out])
}

#[cfg(test)]
mod tests {
    use super::*;
    use hugr::HugrView;

    #[test]
    fn test_barrier_op_factory_creation() {
        let factory = BarrierOperationFactory::new();
        assert_eq!(factory.cache.len(), 0);
    }

    #[test]
    fn test_runtime_barrier() -> Result<(), BuildError> {
        let mut factory = BarrierOperationFactory::new();
        let mut builder = DFGBuilder::new(Signature::new_endo(vec![qb_t(), qb_t(), qb_t()]))?;

        let inputs = builder.input().outputs().collect::<Vec<_>>();
        let outputs = factory.build_runtime_barrier(&mut builder, inputs)?;

        let hugr = builder.finish_hugr_with_outputs(outputs)?;
        assert!(hugr.validate().is_ok());
        Ok(())
    }

    #[test]
    fn test_build_runtime_barrier_op() -> Result<(), BuildError> {
        let array_size = 4;
        let hugr = build_runtime_barrier_op(array_size)?;
        assert!(hugr.validate().is_ok());
        Ok(())
    }
}
