//! Barrier-specific operations that use generic container unpacking/repacking.
//!
//! This module provides barrier-specific functionality that leverages the generic
//! container operations for unpacking and repacking container types, but focuses
//! specifically on qubit extraction and runtime barrier insertion.

use std::sync::Arc;

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

use crate::extension::qsystem::{
    cached_extensions::{ExtensionCache, OpHashWrapper},
    QSystemOpBuilder,
};

/// Factory for creating barrier-specific operations that use generic container operations.
///
/// This factory focuses on barrier-specific functionality like runtime barrier insertion,
/// while delegating generic container unpacking/repacking to the ContainerOperationFactory.
pub struct BarrierOperationFactory {
    /// Temporary extension used for barrier-specific operations.
    extension: Arc<Extension>,
    cache: ExtensionCache,
}

impl BarrierOperationFactory {
    /// Temporary extension name for barrier-specific operations.
    pub(super) const TEMP_EXT_NAME: hugr::hugr::IdentList =
        hugr::hugr::IdentList::new_static_unchecked("__tket.barrier.temp");

    // Barrier-specific operation names.
    pub(super) const WRAPPED_BARRIER: hugr::ops::OpName =
        hugr::ops::OpName::new_static("wrapped_barrier");

    /// Create a new instance of the BarrierOperationFactory.
    pub fn new() -> Self {
        Self {
            extension: Self::build_extension(),
            cache: ExtensionCache::new(),
        }
    }

    /// Gets access to the underlying function cache for operation replacement
    pub fn funcs(&self) -> impl Iterator<Item = (&OpHashWrapper, &Hugr)> {
        self.cache.iter()
    }
    fn build_extension() -> Arc<Extension> {
        Extension::new_arc(
            Self::TEMP_EXT_NAME,
            hugr::extension::Version::new(0, 0, 0),
            |ext, ext_ref| {
                // version of runtime barrier that takes a variable number of qubits
                ext.add_op(
                    Self::WRAPPED_BARRIER,
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
            self.extension
                .get_op(&Self::WRAPPED_BARRIER)
                .unwrap()
                .clone(),
            args.clone(),
        )
        .unwrap();
        self.cache.apply_cached_operation(
            builder,
            op,
            &[TypeArg::BoundedNat(size as u64)],
            qubit_wires,
            |func_b| func_b.build_wrapped_barrier(func_b.input_wires()),
        )
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
        assert_eq!(factory.funcs().count(), 0);
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
