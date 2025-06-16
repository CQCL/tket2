pub(super) mod parser;

use hugr::builder::{Dataflow, FunctionBuilder};
use hugr::{Hugr, Wire};

use crate::extension::rotation::RotationOp;

/// The type of a loaded parameter in the Hugr.
#[derive(Debug, derive_more::Display, Clone, Copy, Hash, PartialEq, Eq)]
pub(super) enum LoadedParameterType {
    Float,
    Rotation,
}

/// A loaded parameter in the Hugr.
///
/// Tracking the type of the wire lets us delay conversion between the types
/// until they are actually needed.
#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub(super) struct LoadedParameter {
    /// The type of the parameter.
    pub typ: LoadedParameterType,
    /// The wire where the parameter is loaded.
    pub wire: Wire,
}

impl LoadedParameter {
    /// Returns a `LoadedParameter` for a float param.
    pub fn float(wire: Wire) -> LoadedParameter {
        LoadedParameter {
            typ: LoadedParameterType::Float,
            wire,
        }
    }

    /// Returns a `LoadedParameter` for a rotation param.
    pub fn rotation(wire: Wire) -> LoadedParameter {
        LoadedParameter {
            typ: LoadedParameterType::Rotation,
            wire,
        }
    }

    /// Convert the parameter into a given type, if necessary.
    ///
    /// Adds the necessary operations to the Hugr and returns a new wire.
    pub fn as_type(
        &self,
        typ: LoadedParameterType,
        hugr: &mut FunctionBuilder<Hugr>,
    ) -> LoadedParameter {
        match (self.typ, typ) {
            (LoadedParameterType::Float, LoadedParameterType::Rotation) => {
                let wire = hugr
                    .add_dataflow_op(RotationOp::from_halfturns_unchecked, [self.wire])
                    .unwrap()
                    .out_wire(0);
                LoadedParameter::rotation(wire)
            }
            (LoadedParameterType::Rotation, LoadedParameterType::Float) => {
                let wire = hugr
                    .add_dataflow_op(RotationOp::to_halfturns, [self.wire])
                    .unwrap()
                    .out_wire(0);
                LoadedParameter::float(wire)
            }
            _ => {
                debug_assert_eq!(self.typ, typ, "cannot convert {} to {}", self.typ, typ);
                *self
            }
        }
    }

    /// Convert the parameter into a float, if necessary.
    ///
    /// Adds the necessary operations to the Hugr and returns a new wire.
    pub fn as_float(&self, hugr: &mut FunctionBuilder<Hugr>) -> LoadedParameter {
        self.as_type(LoadedParameterType::Float, hugr)
    }

    /// Convert the parameter into a rotation, if necessary.
    ///
    /// Adds the necessary operations to the Hugr and returns a new wire.
    pub fn as_rotation(&self, hugr: &mut FunctionBuilder<Hugr>) -> LoadedParameter {
        self.as_type(LoadedParameterType::Rotation, hugr)
    }
}
