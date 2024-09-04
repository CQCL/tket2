//! This module defines the Hugr extension used to represent H-series
//! quantum operations.
//!
//! In the case of lazy operations,
//! laziness is represented by returning `tket2.futures.Future` classical
//! values. Qubits are never lazy.
use hugr::{
    builder::{BuildError, Dataflow},
    extension::{
        prelude::{BOOL_T, QB_T},
        simple_op::{try_from_name, MakeOpDef, MakeRegisteredOp},
        ExtensionId, ExtensionRegistry, OpDef, SignatureFunc, Version, PRELUDE,
    },
    std_extensions::arithmetic::float_types::{EXTENSION as FLOAT_TYPES, FLOAT64_TYPE},
    type_row,
    types::Signature,
    Extension, Wire,
};

use lazy_static::lazy_static;
use strum_macros::{EnumIter, EnumString, IntoStaticStr};
use tket2::extension::TKET2_EXTENSION;

use crate::extension::futures;

use super::futures::future_type;

mod lower;
pub use lower::lower_tk2_op;
use lower::pi_mul;

/// The "tket2.hseries" extension id.
pub const EXTENSION_ID: ExtensionId = ExtensionId::new_unchecked("tket2.hseries");
/// The "tket2.hseries" extension version.
pub const EXTENSION_VERSION: Version = Version::new(0, 1, 0);

lazy_static! {
    /// The "tket2.hseries" extension.
    pub static ref EXTENSION: Extension = {
        let mut ext = Extension::new(EXTENSION_ID, EXTENSION_VERSION);
        HSeriesOp::load_all_ops(&mut ext).unwrap();
        ext
    };

    /// Extension registry including the "tket2.hseries" extension and
    /// dependencies.
    pub static ref REGISTRY: ExtensionRegistry = ExtensionRegistry::try_new([
        futures::EXTENSION.to_owned(),
        PRELUDE.to_owned(),
        EXTENSION.to_owned(),
        FLOAT_TYPES.to_owned(),
        TKET2_EXTENSION.to_owned()
    ]).unwrap();
}

#[derive(
    Clone,
    Copy,
    Debug,
    serde::Serialize,
    serde::Deserialize,
    Hash,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    EnumIter,
    IntoStaticStr,
    EnumString,
)]
#[allow(missing_docs)]
#[non_exhaustive]
pub enum HSeriesOp {
    Measure,
    LazyMeasure,
    Rz,
    PhasedX,
    ZZMax,
    ZZPhase,
    QAlloc,
    QFree,
    Reset,
}

impl MakeOpDef for HSeriesOp {
    fn signature(&self) -> SignatureFunc {
        use HSeriesOp::*;
        let one_qb_row = type_row![QB_T];
        let two_qb_row = type_row![QB_T, QB_T];
        match self {
            LazyMeasure => Signature::new(QB_T, vec![QB_T, future_type(BOOL_T)]),
            Reset => Signature::new(one_qb_row.clone(), one_qb_row),
            ZZMax => Signature::new(two_qb_row.clone(), two_qb_row),
            ZZPhase => Signature::new(type_row![QB_T, QB_T, FLOAT64_TYPE], two_qb_row),
            Measure => Signature::new(one_qb_row, type_row![QB_T, BOOL_T]),
            Rz => Signature::new(type_row![QB_T, FLOAT64_TYPE], one_qb_row),
            PhasedX => Signature::new(type_row![QB_T, FLOAT64_TYPE, FLOAT64_TYPE], one_qb_row),
            QAlloc => Signature::new(type_row![], one_qb_row),
            QFree => Signature::new(one_qb_row, type_row![]),
        }
        .into()
    }

    fn from_def(op_def: &OpDef) -> Result<Self, hugr::extension::simple_op::OpLoadError> {
        try_from_name(op_def.name(), op_def.extension())
    }

    fn extension(&self) -> ExtensionId {
        EXTENSION_ID
    }
}

impl MakeRegisteredOp for HSeriesOp {
    fn extension_id(&self) -> ExtensionId {
        EXTENSION_ID
    }

    fn registry<'s, 'r: 's>(&'s self) -> &'r ExtensionRegistry {
        &REGISTRY
    }
}

/// An extension trait for [Dataflow] providing methods to add
/// "tket2.hseries" operations.
pub trait HSeriesOpBuilder: Dataflow {
    /// Add a "tket2.hseries.LazyMeasure" op.
    fn add_lazy_measure(&mut self, qb: Wire) -> Result<[Wire; 2], BuildError> {
        Ok(self
            .add_dataflow_op(HSeriesOp::LazyMeasure, [qb])?
            .outputs_arr())
    }

    /// Add a "tket2.hseries.Measure" op.
    fn add_measure(&mut self, qb: Wire) -> Result<[Wire; 2], BuildError> {
        Ok(self
            .add_dataflow_op(HSeriesOp::Measure, [qb])?
            .outputs_arr())
    }

    /// Add a "tket2.hseries.Reset" op.
    fn add_reset(&mut self, qb: Wire) -> Result<Wire, BuildError> {
        Ok(self
            .add_dataflow_op(HSeriesOp::Reset, [qb])?
            .outputs()
            .next()
            .unwrap())
    }

    /// Add a "tket2.hseries.ZZMax" op.
    fn add_zz_max(&mut self, qb1: Wire, qb2: Wire) -> Result<[Wire; 2], BuildError> {
        Ok(self
            .add_dataflow_op(HSeriesOp::ZZMax, [qb1, qb2])?
            .outputs_arr())
    }

    /// Add a "tket2.hseries.ZZPhase" op.
    fn add_zz_phase(&mut self, qb1: Wire, qb2: Wire, angle: Wire) -> Result<[Wire; 2], BuildError> {
        Ok(self
            .add_dataflow_op(HSeriesOp::ZZPhase, [qb1, qb2, angle])?
            .outputs_arr())
    }

    /// Add a "tket2.hseries.PhasedX" op.
    fn add_phased_x(&mut self, qb: Wire, angle1: Wire, angle2: Wire) -> Result<Wire, BuildError> {
        Ok(self
            .add_dataflow_op(HSeriesOp::PhasedX, [qb, angle1, angle2])?
            .outputs()
            .next()
            .unwrap())
    }

    /// Add a "tket2.hseries.Rz" op.
    fn add_rz(&mut self, qb: Wire, angle: Wire) -> Result<Wire, BuildError> {
        Ok(self
            .add_dataflow_op(HSeriesOp::Rz, [qb, angle])?
            .outputs()
            .next()
            .unwrap())
    }

    /// Add a "tket2.hseries.QAlloc" op.
    fn add_qalloc(&mut self) -> Result<Wire, BuildError> {
        Ok(self
            .add_dataflow_op(HSeriesOp::QAlloc, [])?
            .outputs()
            .next()
            .unwrap())
    }

    /// Add a "tket2.hseries.QFree" op.
    fn add_qfree(&mut self, qb: Wire) -> Result<(), BuildError> {
        self.add_dataflow_op(HSeriesOp::QFree, [qb])?;
        Ok(())
    }

    /// Build a hadamard gate in terms of HSeries primitives.
    fn build_h(&mut self, qb: Wire) -> Result<Wire, BuildError>
    where
        Self: Sized,
    {
        let pi = pi_mul(self, 1.0);
        let pi_2 = pi_mul(self, 0.5);
        let pi_minus_2 = pi_mul(self, -0.5);

        let q = self.add_phased_x(qb, pi_2, pi_minus_2)?;
        self.add_rz(q, pi)
    }
}

impl<D: Dataflow> HSeriesOpBuilder for D {}

#[cfg(test)]
mod test {
    use std::sync::Arc;

    use cool_asserts::assert_matches;
    use futures::FutureOpBuilder as _;
    use hugr::builder::{DataflowHugr, FunctionBuilder};
    use hugr::ops::{NamedOp, OpType};
    use strum::IntoEnumIterator as _;

    use super::*;

    fn get_opdef(op: impl NamedOp) -> Option<&'static Arc<OpDef>> {
        EXTENSION.get_op(&op.name())
    }

    #[test]
    fn create_extension() {
        assert_eq!(EXTENSION.name(), &EXTENSION_ID);

        for o in HSeriesOp::iter() {
            assert_eq!(HSeriesOp::from_def(get_opdef(o).unwrap()), Ok(o));
        }
    }

    #[test]
    fn lazy_circuit() {
        let hugr = {
            let mut func_builder =
                FunctionBuilder::new("circuit", Signature::new(QB_T, vec![QB_T, BOOL_T])).unwrap();
            let [qb] = func_builder.input_wires_arr();
            let [qb, lazy_b] = func_builder.add_lazy_measure(qb).unwrap();
            let [b] = func_builder.add_read(lazy_b, BOOL_T).unwrap();
            func_builder
                .finish_hugr_with_outputs([qb, b], &REGISTRY)
                .unwrap()
        };
        assert_matches!(hugr.validate(&REGISTRY), Ok(_));
    }

    #[test]
    fn all_ops() {
        let hugr = {
            let mut func_builder = FunctionBuilder::new(
                "all_ops",
                Signature::new(vec![QB_T, FLOAT64_TYPE], vec![QB_T, BOOL_T]),
            )
            .unwrap();
            let [q0, angle] = func_builder.input_wires_arr();
            let q1 = func_builder.add_qalloc().unwrap();
            let q0 = func_builder.add_reset(q0).unwrap();
            let q1 = func_builder.add_phased_x(q1, angle, angle).unwrap();
            let [q0, q1] = func_builder.add_zz_max(q0, q1).unwrap();
            let [q0, q1] = func_builder.add_zz_phase(q0, q1, angle).unwrap();
            let q0 = func_builder.add_rz(q0, angle).unwrap();
            let [q0, b] = func_builder.add_measure(q0).unwrap();
            func_builder.add_qfree(q1).unwrap();
            func_builder
                .finish_hugr_with_outputs([q0, b], &REGISTRY)
                .unwrap()
        };
        assert_matches!(hugr.validate(&REGISTRY), Ok(_));
    }

    #[test]
    fn test_cast() {
        // test overlapping names don't cause cast errors
        for op in HSeriesOp::iter() {
            let optype: OpType = op.into();
            let new_op: HSeriesOp = optype.cast().unwrap();
            assert_eq!(op, new_op);
            assert_eq!(optype.cast::<tket2::Tk2Op>(), None);
        }
    }
}
