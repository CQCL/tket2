//! Centralized temporary extension and cached function management.
//!
//! This module provides a unified framework for creating temporary extensions and caching
//! function definitions across different operation factories. This eliminates duplication
//! of the temporary extension + cached function mapping logic.

use hugr::{
    algorithms::mangle_name,
    builder::{BuildError, Dataflow, DataflowHugr, FunctionBuilder},
    ops::{DataflowOpTrait, ExtensionOp},
    types::TypeArg,
    Hugr, Wire,
};
use indexmap::IndexMap;
use std::ops::Deref;

/// Wrapper for ExtensionOp that implements Hash
#[derive(Clone, PartialEq, Eq)]
pub struct OpHashWrapper(ExtensionOp);

impl From<ExtensionOp> for OpHashWrapper {
    fn from(op: ExtensionOp) -> Self {
        Self(op)
    }
}

impl OpHashWrapper {
    pub fn extension_op(&self) -> &ExtensionOp {
        &self.0
    }
}

impl std::hash::Hash for OpHashWrapper {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.0.extension_id().hash(state);
        self.0.unqualified_id().hash(state);
        self.0.args().hash(state);
    }
}

/// Centralized cache for temporary extensions and their function definitions.
///
/// This provides a single point for managing temporary extensions and their cached
/// function implementations across different operation factories.
pub struct ExtensionCache {
    /// Cached function definitions for each operation instance
    funcs: IndexMap<OpHashWrapper, Hugr>,
}

impl ExtensionCache {
    /// Create a new empty cache
    pub fn new() -> Self {
        Self {
            funcs: IndexMap::new(),
        }
    }

    /// Apply a cached operation, creating and caching the function definition if needed
    pub fn apply_cached_operation<I, F>(
        &mut self,
        builder: &mut impl Dataflow,
        op: ExtensionOp,
        mangle_args: &[TypeArg],
        inputs: I,
        func_builder: F,
    ) -> Result<hugr::builder::handle::Outputs, BuildError>
    where
        I: IntoIterator<Item = Wire>,
        F: FnOnce(&mut FunctionBuilder<Hugr>) -> Result<Vec<Wire>, BuildError>,
    {
        self.cache_function(&op, mangle_args, func_builder)?;
        Ok(builder.add_dataflow_op(op, inputs)?.outputs())
    }

    /// Cache a function definition for a given operation
    pub fn cache_function<F>(
        &mut self,
        op: &ExtensionOp,
        mangle_args: &[TypeArg],
        func_builder: F,
    ) -> Result<(), BuildError>
    where
        F: FnOnce(&mut FunctionBuilder<Hugr>) -> Result<Vec<Wire>, BuildError>,
    {
        let key = OpHashWrapper::from(op.clone());
        // clippy's suggested fix does not make the borrow checker happy
        #[allow(clippy::map_entry)]
        if !self.funcs.contains_key(&key) {
            let name = mangle_name(op.def().name(), mangle_args);
            let sig = op.signature().deref().clone();
            let mut func_b = FunctionBuilder::new(name, sig)?;
            let outputs = func_builder(&mut func_b)?;
            let hugr = func_b.finish_hugr_with_outputs(outputs)?;

            self.funcs.insert(key, hugr);
        }
        Ok(())
    }

    pub fn iter(&self) -> impl Iterator<Item = (&OpHashWrapper, &Hugr)> {
        self.funcs.iter()
    }
}

impl Default for ExtensionCache {
    fn default() -> Self {
        Self::new()
    }
}

impl IntoIterator for ExtensionCache {
    type Item = (OpHashWrapper, Hugr);
    type IntoIter = indexmap::map::IntoIter<OpHashWrapper, Hugr>;

    fn into_iter(self) -> Self::IntoIter {
        self.funcs.into_iter()
    }
}
