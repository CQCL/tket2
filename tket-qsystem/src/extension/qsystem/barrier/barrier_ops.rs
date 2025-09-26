//! Barrier-specific operations that use generic container unpacking/repacking.
//!
//! This module provides barrier-specific functionality that leverages the generic
//! container operations for unpacking and repacking container types, but focuses
//! specifically on qubit extraction and runtime barrier insertion.

use std::sync::Arc;

use hugr::{
    algorithms::mangle_name,
    builder::{BuildError, DFGBuilder, Dataflow, DataflowHugr, FunctionBuilder},
    extension::{prelude::qb_t, Extension},
    ops::{DataflowOpTrait, ExtensionOp, OpName},
    std_extensions::collections::array::array_type,
    types::{
        type_param::TypeParam, FuncValueType, PolyFuncTypeRV, Signature, TypeArg, TypeBound, TypeRV,
    },
    Hugr, Wire,
};

use tket::analysis::type_unpack::TypeUnpacker;

use crate::extension::qsystem::{
    cached_extensions::{ExtensionCache, OpHashWrapper},
    container::ContainerOperationFactory,
    QSystemOpBuilder,
};

/// Factory for creating barrier-specific operations that use generic container operations.
///
/// This factory focuses on barrier-specific functionality like runtime barrier insertion,
/// while delegating generic container unpacking/repacking to the ContainerOperationFactory.
pub struct BarrierOperationFactory {
    /// Container operation factory for generic unpacking/repacking
    pub(super) container_factory: ContainerOperationFactory,
    /// Temporary extension used for barrier-specific operations.
    extension: Arc<Extension>,
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
            container_factory: ContainerOperationFactory::new(TypeUnpacker::for_qubits()),
            extension: Self::build_extension(),
        }
    }

    /// Gets a reference to the internal type analyzer
    pub fn type_analyzer(&mut self) -> &mut TypeUnpacker {
        self.container_factory.type_analyzer()
    }

    /// Gets access to the underlying function cache for operation replacement
    pub fn funcs(&self) -> &indexmap::IndexMap<OpHashWrapper, Hugr> {
        self.container_factory.funcs()
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

    /// Apply a cached operation, creating and caching the function definition if needed
    pub fn apply_cached_operation<I, F>(
        &mut self,
        builder: &mut impl Dataflow,
        op_name: &OpName,
        args: impl Clone + Into<Vec<TypeArg>>,
        mangle_args: &[TypeArg],
        inputs: I,
        func_builder: F,
    ) -> Result<hugr::builder::handle::Outputs, BuildError>
    where
        I: IntoIterator<Item = Wire>,
        F: FnOnce(&mut Self, &mut FunctionBuilder<Hugr>) -> Result<Vec<Wire>, BuildError>,
    {
        let op = ExtensionOp::new(
            self.extension.get_op(op_name).unwrap().clone(),
            args.clone(),
        )
        .unwrap();

        self.cache_function(&op, mangle_args, func_builder)?;
        Ok(builder.add_dataflow_op(op, inputs)?.outputs())
    }

    /// Cache a function definition for a given operation using external cache
    pub fn cache_function_with_external_cache<F>(
        &mut self,
        cache: &mut ExtensionCache,
        op: &ExtensionOp,
        mangle_args: &[TypeArg],
        func_builder: F,
    ) -> Result<(), BuildError>
    where
        F: FnOnce(&mut Self, &mut FunctionBuilder<Hugr>) -> Result<Vec<Wire>, BuildError>,
    {
        cache.cache_function(op, mangle_args, |func_b| func_builder(self, func_b))
    }

    /// Cache a function definition for a given operation (original method - still uses container cache as fallback)
    pub fn cache_function<F>(
        &mut self,
        op: &ExtensionOp,
        mangle_args: &[TypeArg],
        func_builder: F,
    ) -> Result<(), BuildError>
    where
        F: FnOnce(&mut Self, &mut FunctionBuilder<Hugr>) -> Result<Vec<Wire>, BuildError>,
    {
        use std::ops::Deref;

        let key = OpHashWrapper::from(op.clone());
        // clippy's suggested fix does not make the borrow checker happy
        #[allow(clippy::map_entry)]
        if !self.container_factory.funcs().contains_key(&key) {
            let name = mangle_name(op.def().name(), mangle_args);
            let sig = op.signature().deref().clone();
            let mut func_b = FunctionBuilder::new(name, sig)?;
            let outputs = func_builder(self, &mut func_b)?;
            let hugr = func_b.finish_hugr_with_outputs(outputs)?;

            // Insert into the container factory's cache since that's what the barrier inserter expects
            self.container_factory.funcs.insert(key, hugr);
        }
        Ok(())
    }

    /// Build a runtime barrier across the given qubit wires using external cache
    pub fn build_runtime_barrier_with_cache(
        &mut self,
        cache: &mut ExtensionCache,
        builder: &mut impl Dataflow,
        qubit_wires: Vec<Wire>,
    ) -> Result<hugr::builder::handle::Outputs, BuildError> {
        let size = qubit_wires.len();
        let qb_row = vec![qb_t(); size];
        let args = [TypeArg::List(
            qb_row.clone().into_iter().map(Into::into).collect(),
        )];

        cache.apply_cached_operation(
            builder,
            &self.extension,
            &Self::WRAPPED_BARRIER,
            args,
            &[TypeArg::BoundedNat(size as u64)],
            qubit_wires,
            |func_b| func_b.build_wrapped_barrier(func_b.input_wires()),
        )
    }

    /// Build a runtime barrier across the given qubit wires (original method)
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

        self.apply_cached_operation(
            builder,
            &Self::WRAPPED_BARRIER,
            args,
            &[TypeArg::BoundedNat(size as u64)],
            qubit_wires,
            |_, func_b| func_b.build_wrapped_barrier(func_b.input_wires()),
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
        assert_eq!(factory.funcs().len(), 0);
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
