//! Indexing scheme into StaticSizeCircuit.

mod pattern;

pub use pattern::{DisconnectedCircuit, PatternOpPosition};

use std::collections::{BTreeMap, VecDeque};

pub(crate) use pattern::CircuitPath;
use portmatching::indexing as pmx;

use crate::static_circ::{OpPosition, StaticSizeCircuit};

/// Indexing scheme for `StaticSizeCircuit`.
#[derive(Clone, Copy, Default, serde::Serialize, serde::Deserialize)]
pub struct StaticIndexScheme;

/// A map taking pairs (K, isize) as keys, where the isize is expected to
/// be within a contiguous interval of indices.
#[derive(Clone, Debug)]
pub struct OpLocationMap<K, V>(BTreeMap<K, (usize, VecDeque<Option<V>>)>);

impl<K: Ord, V: Clone> OpLocationMap<K, V> {
    pub(crate) fn get_val(&self, key: &K, idx: isize) -> Option<&V> {
        let (offset, vec) = self.0.get(key)?;
        let idx = offset.checked_add_signed(idx)?;
        vec.get(idx)?.as_ref()
    }

    pub(crate) fn set_val(&mut self, key: K, idx: isize, val: V) {
        let (offset, vec) = self.0.entry(key).or_default();
        while offset.checked_add_signed(idx).is_none() {
            vec.push_front(None);
            *offset += 1;
        }
        let idx = offset.checked_add_signed(idx).unwrap();
        if vec.len() <= idx {
            vec.resize(idx + 1, None);
        }
        vec[idx] = Some(val);
    }

    pub(crate) fn values(&self) -> impl Iterator<Item = &V> {
        self.0
            .values()
            .flat_map(|(_, vec)| vec.iter().filter_map(|v| v.as_ref()))
    }
}

impl<K, V: Clone> Default for OpLocationMap<K, V> {
    fn default() -> Self {
        Self(BTreeMap::new())
    }
}

impl<V: pmx::IndexValue> pmx::IndexMap for OpLocationMap<CircuitPath, V> {
    type Key = PatternOpPosition;

    type Value = V;

    type ValueRef<'a> = &'a V
    where
        Self: 'a;

    fn get(&self, var: &Self::Key) -> Option<Self::ValueRef<'_>> {
        let PatternOpPosition { qubit, op_idx } = var;
        self.get_val(qubit, *op_idx as isize)
    }

    fn bind(&mut self, var: Self::Key, val: Self::Value) -> Result<(), pmx::BindVariableError> {
        if let Some(curr_value) = self.get(&var) {
            return Err(pmx::BindVariableError::VariableExists {
                key: format!("{:?}", var),
                curr_value: format!("{:?}", curr_value),
                new_value: format!("{:?}", val),
            });
        }

        let PatternOpPosition { qubit, op_idx } = var;
        self.set_val(qubit, op_idx as isize, val);
        Ok(())
    }
}

impl pmx::IndexingScheme<StaticSizeCircuit> for StaticIndexScheme {
    type Map = OpLocationMap<CircuitPath, OpPosition>;

    fn valid_bindings(
        &self,
        key: &PatternOpPosition,
        known_bindings: &Self::Map,
        data: &StaticSizeCircuit,
    ) -> pmx::BindingResult<Self, StaticSizeCircuit> {
        let get_known = |key| <Self::Map as pmx::IndexMap>::get(known_bindings, key);
        if let Some(v) = <Self::Map as pmx::IndexMap>::get(known_bindings, key) {
            // Already bound.
            Ok(vec![*v].into())
        } else if key.op_idx != 0 {
            // Can only bind if the idx 0 is bound.
            if let Some(root) = get_known(&key.with_op_idx(0)) {
                let Some(pos) = root.try_add_op_idx(key.op_idx as isize) else {
                    return Ok(vec![].into());
                };
                if data.exists(pos) {
                    Ok(vec![pos].into())
                } else {
                    Ok(vec![].into())
                }
            } else {
                Err(pmx::MissingIndexKeys(vec![key.with_op_idx(0)]))
            }
        } else {
            // Bind first op on a new qubit
            if key.qubit.is_empty() {
                // It is the root of the pattern, all locations are valid
                Ok(Vec::from_iter(data.positions_iter()).into())
            } else {
                // It is a new qubit, use the root to resolve it.
                if let Some(&root) = get_known(&PatternOpPosition::root()) {
                    Ok(Vec::from_iter(key.resolve(data, root)).into())
                } else {
                    Err(pmx::MissingIndexKeys(vec![PatternOpPosition::root()]))
                }
            }
        }
    }
}
