//! Mapping from extension operation instances to function definitions
//! that can be used to replace them.
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

/// Map from extension operation instances to a function definition
/// that can be used to replace it.
pub struct OpFunctionMap {
    // RefCell for interior mutability, allowing
    // recursive function building.
    map: RefCell<IndexMap<OpHashWrapper, Hugr>>,
}

impl OpFunctionMap {
    /// Create a new empty map
    pub fn new() -> Self {
        Self {
            map: RefCell::new(IndexMap::new()),
        }
    }

    /// Insert a function definition for the given operation instance,
    /// if it is not already present.
    pub fn insert_with<O, F>(
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
        if self.map.borrow().contains_key(&key) {
            return Ok(());
        }
        let name = mangle_name(op.def().name(), mangle_args);
        let sig = op.signature().deref().clone();
        let mut func_b = FunctionBuilder::new(name, sig)?;
        let outputs = func_builder(&mut func_b)?;
        let hugr = func_b.finish_hugr_with_outputs(outputs)?;
        self.map.borrow_mut().insert(key, hugr);
        Ok(())
    }

    pub fn into_iter(self) -> impl Iterator<Item = (ExtensionOp, Hugr)> {
        self.map.into_inner().into_iter().map(|(k, v)| (k.0, v))
    }
}

impl Default for OpFunctionMap {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    impl super::OpFunctionMap {
        /// Return the number of stored functions
        pub fn len(&self) -> usize {
            self.map.borrow().len()
        }
    }
}
