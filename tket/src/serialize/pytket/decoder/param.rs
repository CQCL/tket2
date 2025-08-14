//! Definition of a loaded parameter (either floating point or a rotation type) attached to a HUGR wire.
pub(super) mod parser;
use std::sync::LazyLock;

use hugr::builder::{Dataflow, FunctionBuilder};
use hugr::std_extensions::arithmetic::float_types::float64_type;
use hugr::types::Type;
use hugr::{Hugr, Wire};

use crate::extension::rotation::{rotation_type, RotationOp};

/// The type of a loaded parameter in the Hugr.
#[derive(Debug, derive_more::Display, Clone, Copy, Hash, PartialEq, Eq)]
pub enum LoadedParameterType {
    /// A float parameter.
    Float,
    /// A rotation parameter.
    Rotation,
}

/// A loaded parameter in the Hugr.
///
/// Tracking the type of the wire lets us delay conversion between the types
/// until they are actually needed.
#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub struct LoadedParameter {
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

    /// Returns the hugr type for the parameter.
    #[expect(
        dead_code,
        reason = "Temporarily unused while we refactor the pytket decoder"
    )]
    pub fn wire_type(&self) -> &Type {
        static FLOAT_TYPE: LazyLock<Type> = LazyLock::new(float64_type);
        static ROTATION_TYPE: LazyLock<Type> = LazyLock::new(rotation_type);
        match self.typ {
            LoadedParameterType::Float => &FLOAT_TYPE,
            LoadedParameterType::Rotation => &ROTATION_TYPE,
        }
    }

    /// Convert the parameter into a given type, if necessary.
    ///
    /// Adds the necessary operations to the Hugr and returns a new wire.
    ///
    /// See [`LoadedParameter::as_float`] and [`LoadedParameter::as_rotation`]
    /// for more convenient methods.
    pub fn with_type<H: AsRef<Hugr> + AsMut<Hugr>>(
        &self,
        typ: LoadedParameterType,
        hugr: &mut FunctionBuilder<H>,
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
    pub fn as_float<H: AsRef<Hugr> + AsMut<Hugr>>(
        &self,
        hugr: &mut FunctionBuilder<H>,
    ) -> LoadedParameter {
        self.with_type(LoadedParameterType::Float, hugr)
    }

    /// Convert the parameter into a rotation, if necessary.
    ///
    /// Adds the necessary operations to the Hugr and returns a new wire.
    pub fn as_rotation<H: AsRef<Hugr> + AsMut<Hugr>>(
        &self,
        hugr: &mut FunctionBuilder<H>,
    ) -> LoadedParameter {
        self.with_type(LoadedParameterType::Rotation, hugr)
    }
}
