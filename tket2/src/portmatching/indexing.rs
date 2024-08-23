mod pattern;

pub use pattern::PatternOpLocation;

use std::collections::{BTreeMap, VecDeque};

use pattern::CircuitPath;
use portmatching::indexing as pmx;

use crate::static_circ::{OpLocation, StaticSizeCircuit};

/// Indexing scheme for `StaticSizeCircuit`.
#[derive(Clone, Copy, Default, serde::Serialize, serde::Deserialize)]
pub struct StaticIndexScheme;

/// A 2d map taking `PatternOpLocation`s as keys.
#[derive(Clone)]
pub struct Map<V>(BTreeMap<CircuitPath, (usize, VecDeque<Option<V>>)>);

impl<V: Clone> Default for Map<V> {
    fn default() -> Self {
        Self(BTreeMap::new())
    }
}

impl<V: pmx::IndexValue> pmx::IndexMap for Map<V> {
    type Key = PatternOpLocation;

    type Value = V;

    type ValueRef<'a> = &'a V
    where
        Self: 'a;

    fn get(&self, var: &Self::Key) -> Option<Self::ValueRef<'_>> {
        let PatternOpLocation { qubit, op_idx } = var;
        let (offset, vec) = self.0.get(qubit)?;
        let idx = offset.checked_add_signed(*op_idx as isize)?;
        vec.get(idx)?.as_ref()
    }

    fn bind(&mut self, var: Self::Key, val: Self::Value) -> Result<(), pmx::BindVariableError> {
        if let Some(curr_value) = self.get(&var) {
            return Err(pmx::BindVariableError::VariableExists {
                key: format!("{:?}", var),
                curr_value: format!("{:?}", curr_value),
                new_value: format!("{:?}", val),
            });
        }

        let PatternOpLocation { qubit, op_idx } = var;
        let (offset, vec) = self.0.entry(qubit).or_default();
        while offset.checked_add_signed(op_idx as isize).is_none() {
            vec.push_front(None);
            *offset += 1;
        }
        let idx = offset.checked_add_signed(op_idx as isize).unwrap();
        if vec.len() <= idx {
            vec.resize(idx + 1, None);
        }
        vec[idx] = Some(val);
        Ok(())
    }
}

impl pmx::IndexingScheme<StaticSizeCircuit> for StaticIndexScheme {
    type Map = Map<OpLocation>;

    fn valid_bindings(
        &self,
        key: &pmx::Key<Self, StaticSizeCircuit>,
        known_bindings: &Self::Map,
        data: &StaticSizeCircuit,
    ) -> pmx::BindingResult<Self, StaticSizeCircuit> {
        let get_known = |key| <Self::Map as pmx::IndexMap>::get(known_bindings, key);
        if let Some(v) = <Self::Map as pmx::IndexMap>::get(known_bindings, key) {
            // Already bound.
            Ok(vec![v.clone()].into())
        } else if key.op_idx != 0 {
            // Can only bind if the idx 0 is bound.
            if let Some(root) = get_known(&key.with_op_idx(0)) {
                dbg!(&root);
                let Some(loc) = root.try_add_op_idx(key.op_idx as isize) else {
                    return Ok(vec![].into());
                };
                if data.get(loc).is_some() {
                    Ok(vec![loc].into())
                } else {
                    Ok(vec![].into())
                }
            } else {
                Err(pmx::MissingIndexKeys(vec![key.with_op_idx(0)]))
            }
        } else {
            // Bind first op on a new qubit
            if key.qubit.is_root() {
                // It is the root of the pattern, all locations are valid
                Ok(Vec::from_iter(data.all_locations()).into())
            } else {
                // It is a new qubit, use the root to resolve it.
                if let Some(&root) = get_known(&PatternOpLocation::root()) {
                    Ok(Vec::from_iter(key.resolve(data, root)).into())
                } else {
                    Err(pmx::MissingIndexKeys(vec![PatternOpLocation::root()]))
                }
            }
        }
    }
}
