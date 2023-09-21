//! Filters for the [`Units`] iterator that unwrap the yielded units when
//! possible.

use hugr::extension::prelude;
use hugr::hugr::CircuitUnit;
use hugr::types::Type;
use hugr::{Port, Wire};

use super::{DefaultUnitLabeller, LinearUnit, Units};

/// A filtered units iterator
pub type FilteredUnits<F, UL = DefaultUnitLabeller> = std::iter::FilterMap<
    Units<UL>,
    fn((CircuitUnit, Port, Type)) -> Option<<F as UnitFilter>::Item>,
>;

/// A filter over a [`Units`] iterator.
pub trait UnitFilter {
    /// The item yielded by the filtered iterator.
    type Item;

    /// Filter a [`Units`] iterator item, and unwrap it into a `Self::Item` if
    /// it's accepted.
    fn accept(item: (CircuitUnit, Port, Type)) -> Option<Self::Item>;
}

/// A unit filter that accepts linear units.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Linear;

/// A unit filter that accepts qubits, a subset of [`Linear`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Qubits;

/// A unit filter that accepts non-linear units.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct NonLinear;

impl UnitFilter for Linear {
    type Item = (LinearUnit, Port, Type);

    fn accept(item: (CircuitUnit, Port, Type)) -> Option<Self::Item> {
        match item {
            (CircuitUnit::Linear(unit), port, typ) => Some((unit, port, typ)),
            _ => None,
        }
    }
}

impl UnitFilter for Qubits {
    type Item = (LinearUnit, Port, Type);

    fn accept(item: (CircuitUnit, Port, Type)) -> Option<Self::Item> {
        match item {
            (CircuitUnit::Linear(unit), port, typ) if typ == prelude::QB_T => {
                Some((unit, port, typ))
            }
            _ => None,
        }
    }
}

impl UnitFilter for NonLinear {
    type Item = (Wire, Port, Type);

    fn accept(item: (CircuitUnit, Port, Type)) -> Option<Self::Item> {
        match item {
            (CircuitUnit::Wire(wire), port, typ) => Some((wire, port, typ)),
            _ => None,
        }
    }
}
