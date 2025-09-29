//! Centralized temporary extension and cached function management.
//!
//! This module provides a unified framework for creating temporary extensions and caching
//! function definitions across different operation factories. This eliminates duplication
//! of the temporary extension + cached function mapping logic.

use hugr::{
    algorithms::mangle_name,
    builder::{BuildError, DataflowHugr, FunctionBuilder},
    ops::{DataflowOpTrait, ExtensionOp},
    types::TypeArg,
    Hugr, Wire,
};
use indexmap::IndexMap;
use std::{cell::RefCell, ops::Deref};

/// Wrapper for ExtensionOp that implements Hash
#[derive(Clone, PartialEq, Eq)]
struct OpHashWrapper(ExtensionOp);

impl From<ExtensionOp> for OpHashWrapper {
    fn from(op: ExtensionOp) -> Self {
        Self(op)
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
    funcs: RefCell<IndexMap<OpHashWrapper, Hugr>>,
}

impl ExtensionCache {
    /// Create a new empty cache
    pub fn new() -> Self {
        Self {
            funcs: RefCell::new(IndexMap::new()),
        }
    }

    /// Cache a function definition for a given operation
    pub fn cache_function<O, F>(
        &self,
        op: &ExtensionOp,
        mangle_args: &[TypeArg],
        func_builder: F,
    ) -> Result<(), BuildError>
    where
        O: IntoIterator<Item = Wire>,
        F: FnOnce(&mut FunctionBuilder<Hugr>) -> Result<O, BuildError>,
    {
        let key = OpHashWrapper::from(op.clone());
        if self.funcs.borrow().contains_key(&key) {
            return Ok(());
        }
        let name = mangle_name(op.def().name(), mangle_args);
        let sig = op.signature().deref().clone();
        let mut func_b = FunctionBuilder::new(name, sig)?;
        let outputs = func_builder(&mut func_b)?;
        let hugr = func_b.finish_hugr_with_outputs(outputs)?;
        self.funcs.borrow_mut().insert(key, hugr);
        Ok(())
    }

    pub fn contains(&self, op: &ExtensionOp) -> bool {
        let key = OpHashWrapper::from(op.clone());
        self.funcs.borrow().contains_key(&key)
    }

    pub fn replace(&mut self, op: ExtensionOp, hugr: Hugr) -> Option<Hugr> {
        let key = OpHashWrapper::from(op);
        self.funcs.borrow_mut().insert(key, hugr)
    }

    pub fn len(&self) -> usize {
        self.funcs.borrow().len()
    }

    pub fn into_iter(self) -> impl Iterator<Item = (ExtensionOp, Hugr)> {
        self.funcs.into_inner().into_iter().map(|(k, v)| (k.0, v))
    }
}

impl Default for ExtensionCache {
    fn default() -> Self {
        Self::new()
    }
}
