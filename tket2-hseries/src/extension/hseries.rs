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
        ExtensionId, ExtensionRegistry, ExtensionSet, OpDef, SignatureFunc, Version, PRELUDE,
    },
    ops::Value,
    std_extensions::arithmetic::{
        float_ops::FloatOps,
        float_types::{ConstF64, EXTENSION as FLOAT_TYPES, FLOAT64_TYPE},
    },
    type_row,
    types::Signature,
    Extension, Wire,
};

use lazy_static::lazy_static;
use strum_macros::{EnumIter, EnumString, IntoStaticStr};

use crate::extension::futures;

use super::futures::future_type;

mod lower;
use lower::pi_mul_f64;
pub use lower::{check_lowered, lower_tk2_op};

/// The "tket2.hseries" extension id.
pub const EXTENSION_ID: ExtensionId = ExtensionId::new_unchecked("tket2.hseries");
/// The "tket2.hseries" extension version.
pub const EXTENSION_VERSION: Version = Version::new(0, 1, 0);

lazy_static! {
    /// The "tket2.hseries" extension.
    pub static ref EXTENSION: Extension = {
        let mut ext = Extension::new(EXTENSION_ID, EXTENSION_VERSION).with_reqs(ExtensionSet::from_iter([
            futures::EXTENSION.name(),
            PRELUDE.name(),
            FLOAT_TYPES.name(),
            tket2::extension::angle::ANGLE_EXTENSION.name(),
        ].into_iter().cloned()));
        HSeriesOp::load_all_ops(&mut ext).unwrap();
        ext
    };

    /// Extension registry including the "tket2.hseries" extension and
    /// dependencies.
    pub static ref REGISTRY: ExtensionRegistry = ExtensionRegistry::try_new([
        EXTENSION.to_owned(),
        futures::EXTENSION.to_owned(),
        PRELUDE.to_owned(),
        FLOAT_TYPES.to_owned(),
        tket2::extension::angle::ANGLE_EXTENSION.to_owned(),
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
        Ok(self.add_dataflow_op(HSeriesOp::Reset, [qb])?.out_wire(0))
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
            .out_wire(0))
    }

    /// Add a "tket2.hseries.Rz" op.
    fn add_rz(&mut self, qb: Wire, angle: Wire) -> Result<Wire, BuildError> {
        Ok(self
            .add_dataflow_op(HSeriesOp::Rz, [qb, angle])?
            .out_wire(0))
    }

    /// Add a "tket2.hseries.QAlloc" op.
    fn add_qalloc(&mut self) -> Result<Wire, BuildError> {
        Ok(self.add_dataflow_op(HSeriesOp::QAlloc, [])?.out_wire(0))
    }

    /// Add a "tket2.hseries.QFree" op.
    fn add_qfree(&mut self, qb: Wire) -> Result<(), BuildError> {
        self.add_dataflow_op(HSeriesOp::QFree, [qb])?;
        Ok(())
    }

    /// Build a hadamard gate in terms of HSeries primitives.
    fn build_h(&mut self, qb: Wire) -> Result<Wire, BuildError> {
        // Clifford gate: Hadamard
        // gate h() a
        // {
        // PhasedX(pi/2, -pi/2) a;
        // Rz(pi) a;
        // }
        let pi = pi_mul_f64(self, 1.0);
        let pi_2 = pi_mul_f64(self, 0.5);
        let pi_minus_2 = pi_mul_f64(self, -0.5);

        let q = self.add_phased_x(qb, pi_2, pi_minus_2)?;
        self.add_rz(q, pi)
    }

    /// Build an X gate in terms of HSeries primitives.
    fn build_x(&mut self, qb: Wire) -> Result<Wire, BuildError> {
        // Pauli gate: bit-flip
        // gate x() a
        // {
        //    PhasedX(pi, 0) a;
        // }
        let pi = pi_mul_f64(self, 1.0);
        let zero = pi_mul_f64(self, 0.0);
        self.add_phased_x(qb, pi, zero)
    }

    /// Build a Y gate in terms of HSeries primitives.
    fn build_y(&mut self, qb: Wire) -> Result<Wire, BuildError> {
        // Pauli gate: bit and phase flip
        // gate y() a
        // {
        //    PhasedX(pi, pi/2) a;
        // }
        let pi = pi_mul_f64(self, 1.0);
        let pi_2 = pi_mul_f64(self, 0.5);
        self.add_phased_x(qb, pi, pi_2)
    }

    /// Build a Z gate in terms of HSeries primitives.
    fn build_z(&mut self, qb: Wire) -> Result<Wire, BuildError> {
        let pi = pi_mul_f64(self, 1.0);
        self.add_rz(qb, pi)
    }

    /// Build an S gate in terms of HSeries primitives.
    fn build_s(&mut self, qb: Wire) -> Result<Wire, BuildError> {
        let pi_2 = pi_mul_f64(self, 0.5);
        self.add_rz(qb, pi_2)
    }

    /// Build an Sdg gate in terms of HSeries primitives.
    fn build_sdg(&mut self, qb: Wire) -> Result<Wire, BuildError> {
        let pi_minus_2 = pi_mul_f64(self, -0.5);
        self.add_rz(qb, pi_minus_2)
    }

    /// Build a T gate in terms of HSeries primitives.
    fn build_t(&mut self, qb: Wire) -> Result<Wire, BuildError> {
        let pi_4 = pi_mul_f64(self, 0.25);
        self.add_rz(qb, pi_4)
    }

    /// Build a Tdg gate in terms of HSeries primitives.
    fn build_tdg(&mut self, qb: Wire) -> Result<Wire, BuildError> {
        let pi_minus_4 = pi_mul_f64(self, -0.25);
        self.add_rz(qb, pi_minus_4)
    }

    /// Build a CNOT gate in terms of HSeries primitives.
    fn build_cx(&mut self, c: Wire, t: Wire) -> Result<[Wire; 2], BuildError> {
        // Clifford gate: CNOT
        // gate CX() c,t
        // {
        // PhasedX(-pi/2, pi/2) t;
        // ZZ() c, t;
        // Rz(-pi/2) c;
        // PhasedX(pi/2, pi) t;
        // Rz(-pi/2) t;
        // }
        let pi = pi_mul_f64(self, 1.0);
        let pi_2 = pi_mul_f64(self, 0.5);
        let pi_minus_2 = pi_mul_f64(self, -0.5);

        let t = self.add_phased_x(t, pi_minus_2, pi_2)?;
        let [c, t] = self.add_zz_max(c, t)?;
        let c = self.add_rz(c, pi_minus_2)?;
        let t = self.add_phased_x(t, pi_2, pi)?;
        let t = self.add_rz(t, pi_minus_2)?;

        Ok([c, t])
    }

    /// Build a CY gate in terms of HSeries primitives.
    fn build_cy(&mut self, a: Wire, b: Wire) -> Result<[Wire; 2], BuildError> {
        // gate cy() a,b
        // {
        //    sdg b;
        //    cx a,b;
        //    s b;
        // }
        let b = self.build_sdg(b)?;
        let [a, b] = self.build_cx(a, b)?;
        let b = self.build_s(b)?;
        Ok([a, b])
    }

    /// Build a CZ gate in terms of HSeries primitives.
    fn build_cz(&mut self, a: Wire, b: Wire) -> Result<[Wire; 2], BuildError> {
        // gate cz() a,b
        // {
        //    h b;
        //    cx a,b;
        //    h b;
        // }
        let b = self.build_h(b)?;
        let [a, b] = self.build_cx(a, b)?;
        let b = self.build_h(b)?;
        Ok([a, b])
    }

    /// Build a RX gate in terms of HSeries primitives.
    fn build_rx(&mut self, qb: Wire, theta: Wire) -> Result<Wire, BuildError> {
        // Rotation around X-axis
        // gate rx(theta) a
        // {
        //    phased_x(theta, 0) a;
        // }
        let zero = pi_mul_f64(self, 0.0);
        self.add_phased_x(qb, theta, zero)
    }

    /// Build a RY gate in terms of HSeries primitives.
    fn build_ry(&mut self, qb: Wire, theta: Wire) -> Result<Wire, BuildError> {
        // Rotation around Y-axis
        // gate ry(theta) a
        // {
        //    phased_x(theta, pi/2) a;
        // }
        let pi_2 = pi_mul_f64(self, 0.5);
        self.add_phased_x(qb, theta, pi_2)
    }

    /// Build a CRZ gate in terms of HSeries primitives.
    fn build_crz(&mut self, a: Wire, b: Wire, lambda: Wire) -> Result<[Wire; 2], BuildError> {
        // gate crz(lambda) a,b
        // {
        //    ZZPhase(-lambda/2) a, b;
        //    Rz(lambda/2) b;
        // }
        let two = self.add_load_const(Value::from(ConstF64::new(2.0)));
        let lambda_2 = self
            .add_dataflow_op(FloatOps::fdiv, [lambda, two])?
            .out_wire(0);
        let lambda_minus_2 = self
            .add_dataflow_op(FloatOps::fneg, [lambda_2])?
            .out_wire(0);

        let [a, b] = self.add_zz_phase(a, b, lambda_minus_2)?;
        let b = self.add_rz(b, lambda_2)?;
        Ok([a, b])
    }

    /// Build a Toffoli (CCX) gate in terms of HSeries primitives.
    fn build_toffoli(&mut self, a: Wire, b: Wire, c: Wire) -> Result<[Wire; 3], BuildError> {
        // gate ccx() a,b,c
        // {
        //    h c;
        //    cx b,c; tdg c;
        //    cx a,c; t c;
        //    cx b,c; tdg c;
        //    cx a,c; t b; t c; h c;
        //    cx a,b; t a; tdg b;
        //    cx a,b;
        // }
        let c = self.build_h(c)?;
        let [b, c] = self.build_cx(b, c)?;
        let c = self.build_tdg(c)?;
        let [a, c] = self.build_cx(a, c)?;
        let c = self.build_t(c)?;
        let [b, c] = self.build_cx(b, c)?;
        let c = self.build_tdg(c)?;
        let [a, c] = self.build_cx(a, c)?;
        let b = self.build_t(b)?;
        let c = self.build_t(c)?;
        let c = self.build_h(c)?;
        let [a, b] = self.build_cx(a, b)?;
        let a = self.build_t(a)?;
        let b = self.build_tdg(b)?;
        let [a, b] = self.build_cx(a, b)?;
        Ok([a, b, c])
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
