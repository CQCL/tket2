//! Filters for the [`Units`] iterator that unwrap the yielded units when
//! possible.
//!
//! These are meant to be used as a parameter to [`Iterator::filter_map`].
//!
//! [`Units`]: crate::circuit::units::Units

use hugr::extension::prelude;
use hugr::types::Type;
use hugr::CircuitUnit;
use hugr::Wire;

use super::LinearUnit;

/// A unit filter that return only linear units.
pub fn filter_linear<P>(item: (CircuitUnit, P, Type)) -> Option<(LinearUnit, P, Type)> {
    match item {
        (CircuitUnit::Linear(unit), port, typ) => Some((LinearUnit::new(unit), port, typ)),
        _ => None,
    }
}

/// A unit filter that return only qubits, a subset of [`filter_linear`].
pub fn filter_qubit<P>(item: (CircuitUnit, P, Type)) -> Option<(LinearUnit, P, Type)> {
    match item {
        (CircuitUnit::Linear(unit), port, typ) if typ == prelude::QB_T => {
            Some((LinearUnit::new(unit), port, typ))
        }
        _ => None,
    }
}

/// A unit filter that return only non-linear units.
pub fn filter_non_linear<P>(item: (CircuitUnit, P, Type)) -> Option<(Wire, P, Type)> {
    match item {
        (CircuitUnit::Wire(wire), port, typ) => Some((wire, port, typ)),
        _ => None,
    }
}
