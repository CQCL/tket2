//! This module defines the Hugr extension used to represent H-series
//! quantum operations.
//!
//! In the case of lazy operations,
//! laziness is represented by returning `tket.futures.Future` classical
//! values. Qubits are never lazy.
use std::{
    str::FromStr,
    sync::{Arc, Weak},
};

use hugr::{
    builder::{BuildError, Dataflow, DataflowSubContainer, SubContainer},
    extension::{
        prelude::{bool_t, option_type, qb_t, UnwrapBuilder},
        simple_op::{try_from_name, MakeOpDef, MakeRegisteredOp},
        ExtensionId, ExtensionRegistry, OpDef, SignatureFunc, Version, PRELUDE,
    },
    ops::{ExtensionOp, OpName, Value},
    std_extensions::{
        arithmetic::{
            float_ops::FloatOps,
            float_types::{float64_type, ConstF64, EXTENSION as FLOAT_TYPES},
            int_types::int_type,
        },
        collections::array::{array_type_parametric, ArrayOpBuilder},
    },
    type_row,
    types::{type_param::TypeParam, PolyFuncType, Signature, Type, TypeArg, TypeRow},
    Extension, Wire,
};

use crate::extension::futures;
use derive_more::Display;
use lazy_static::lazy_static;
use strum::{EnumIter, EnumString, IntoStaticStr};
use tket::extension::bool::{bool_type, BoolOp};

use super::futures::future_type;

mod barrier;
mod lower;
use lower::pi_mul_f64;
pub use lower::{check_lowered, lower_tk2_op, LowerTk2Error, LowerTketToQSystemPass};

/// The "tket.qsystem" extension id.
pub const EXTENSION_ID: ExtensionId = ExtensionId::new_unchecked("tket.qsystem");
/// The "tket.qsystem" extension version.
pub const EXTENSION_VERSION: Version = Version::new(0, 5, 0);

lazy_static! {
    /// The "tket.qsystem" extension.
    pub static ref EXTENSION: Arc<Extension> = {
         Extension::new_arc(EXTENSION_ID, EXTENSION_VERSION, |ext, ext_ref| {
            QSystemOp::load_all_ops( ext, ext_ref).unwrap();
            RuntimeBarrierDef.add_to_extension(ext, ext_ref).unwrap();
        })
    };

    /// Extension registry including the "tket.qsystem" extension and
    /// dependencies.
    pub static ref REGISTRY: ExtensionRegistry = ExtensionRegistry::new([
        EXTENSION.to_owned(),
        futures::EXTENSION.to_owned(),
        PRELUDE.to_owned(),
        FLOAT_TYPES.to_owned(),
    ]);
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
    Display,
)]
#[allow(missing_docs)]
#[non_exhaustive]
pub enum QSystemOp {
    Measure,
    LazyMeasure,
    LazyMeasureReset,
    Rz,
    PhasedX,
    ZZPhase,
    TryQAlloc,
    QFree,
    Reset,
    MeasureReset,
    LazyMeasureLeaked,
}

impl MakeOpDef for QSystemOp {
    fn opdef_id(&self) -> hugr::ops::OpName {
        <&'static str>::from(self).into()
    }

    fn init_signature(&self, _extension_ref: &std::sync::Weak<Extension>) -> SignatureFunc {
        use QSystemOp::*;
        let one_qb_row = TypeRow::from(vec![qb_t()]);
        let two_qb_row = TypeRow::from(vec![qb_t(), qb_t()]);
        match self {
            LazyMeasure => Signature::new(qb_t(), future_type(bool_t())),
            LazyMeasureLeaked => Signature::new(qb_t(), future_type(int_type(6))),
            LazyMeasureReset => Signature::new(qb_t(), vec![qb_t(), future_type(bool_t())]),
            Reset => Signature::new(one_qb_row.clone(), one_qb_row),
            ZZPhase => Signature::new(vec![qb_t(), qb_t(), float64_type()], two_qb_row),
            Measure => Signature::new(one_qb_row, bool_type()),
            Rz => Signature::new(vec![qb_t(), float64_type()], one_qb_row),
            PhasedX => Signature::new(vec![qb_t(), float64_type(), float64_type()], one_qb_row),
            TryQAlloc => Signature::new(type_row![], Type::from(option_type(one_qb_row))),
            QFree => Signature::new(one_qb_row, type_row![]),
            MeasureReset => Signature::new(one_qb_row.clone(), vec![qb_t(), bool_type()]),
        }
        .into()
    }

    fn from_def(op_def: &OpDef) -> Result<Self, hugr::extension::simple_op::OpLoadError> {
        try_from_name(op_def.name(), op_def.extension_id())
    }

    fn extension(&self) -> ExtensionId {
        EXTENSION_ID
    }

    fn extension_ref(&self) -> std::sync::Weak<Extension> {
        Arc::downgrade(&EXTENSION)
    }

    fn description(&self) -> String {
        match self {
            QSystemOp::Measure => "Measure a qubit and lose it.",
            QSystemOp::LazyMeasure => "Lazily measure a qubit and lose it.",
            QSystemOp::Rz => "Rotate a qubit around the Z axis. Not physical.",
            QSystemOp::PhasedX => "PhasedX gate.",
            QSystemOp::ZZPhase => "ZZ gate with an angle.",
            QSystemOp::TryQAlloc => "Allocate a qubit in the Z |0> eigenstate.",
            QSystemOp::QFree => "Free a qubit (lose track of it).",
            QSystemOp::Reset => "Reset a qubit to the Z |0> eigenstate.",
            QSystemOp::MeasureReset => "Measure a qubit and reset it to the Z |0> eigenstate.",
            QSystemOp::LazyMeasureLeaked => {
                "Measure a qubit (return 0 or 1) or detect leakage (return 2)."
            }
            QSystemOp::LazyMeasureReset => {
                "Lazily measure a qubit and reset it to the Z |0> eigenstate."
            }
        }
        .to_string()
    }
}

impl MakeRegisteredOp for QSystemOp {
    fn extension_id(&self) -> ExtensionId {
        EXTENSION_ID
    }

    fn extension_ref(&self) -> Weak<Extension> {
        Arc::downgrade(&EXTENSION)
    }
}

/// The name of the "tket.qsystem.RuntimeBarrier" operation.
pub const RUNTIME_BARRIER_NAME: OpName = OpName::new_inline("RuntimeBarrier");

/// Helper struct for the "tket.qsystem.RuntimeBarrier" operation definition.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RuntimeBarrierDef;

impl FromStr for RuntimeBarrierDef {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        if s == RuntimeBarrierDef.opdef_id().as_str() {
            Ok(Self)
        } else {
            Err(())
        }
    }
}

impl MakeOpDef for RuntimeBarrierDef {
    fn from_def(op_def: &OpDef) -> Result<Self, hugr::extension::simple_op::OpLoadError>
    where
        Self: Sized,
    {
        try_from_name(op_def.name(), op_def.extension_id())
    }

    fn extension(&self) -> ExtensionId {
        EXTENSION_ID
    }

    fn extension_ref(&self) -> std::sync::Weak<Extension> {
        Arc::downgrade(&EXTENSION)
    }

    fn init_signature(&self, _extension_ref: &Weak<Extension>) -> SignatureFunc {
        PolyFuncType::new(
            [TypeParam::max_nat_type()],
            Signature::new_endo(
                array_type_parametric(TypeArg::new_var_use(0, TypeParam::max_nat_type()), qb_t())
                    .unwrap(),
            ),
        )
        .into()
    }

    fn description(&self) -> String {
        "Acts as a runtime barrier between operations on argument qubits.".to_string()
    }

    fn opdef_id(&self) -> OpName {
        RUNTIME_BARRIER_NAME
    }
}

/// An extension trait for [Dataflow] providing methods to add
/// "tket.qsystem" operations.
pub trait QSystemOpBuilder: Dataflow + UnwrapBuilder + ArrayOpBuilder {
    /// Add a "tket.qsystem.LazyMeasure" op.
    fn add_lazy_measure(&mut self, qb: Wire) -> Result<Wire, BuildError> {
        Ok(self
            .add_dataflow_op(QSystemOp::LazyMeasure, [qb])?
            .out_wire(0))
    }

    /// Add a "tket.qsystem.LazyMeasureLeaked" op.
    fn add_lazy_measure_leaked(&mut self, qb: Wire) -> Result<Wire, BuildError> {
        Ok(self
            .add_dataflow_op(QSystemOp::LazyMeasureLeaked, [qb])?
            .out_wire(0))
    }

    /// Add a "tket.qsystem.LazyMeasureReset" op.
    fn add_lazy_measure_reset(&mut self, qb: Wire) -> Result<[Wire; 2], BuildError> {
        Ok(self
            .add_dataflow_op(QSystemOp::LazyMeasureReset, [qb])?
            .outputs_arr())
    }

    /// Add a "tket.qsystem.Measure" op.
    fn add_measure(&mut self, qb: Wire) -> Result<Wire, BuildError> {
        Ok(self.add_dataflow_op(QSystemOp::Measure, [qb])?.out_wire(0))
    }

    /// Add a "tket.qsystem.Reset" op.
    fn add_reset(&mut self, qb: Wire) -> Result<Wire, BuildError> {
        Ok(self.add_dataflow_op(QSystemOp::Reset, [qb])?.out_wire(0))
    }

    /// Add a maximally entangling "tket.qsystem.ZZPhase(pi/2)" op.
    fn build_zz_max(&mut self, qb1: Wire, qb2: Wire) -> Result<[Wire; 2], BuildError> {
        let pi_2 = pi_mul_f64(self, 0.5);
        self.add_zz_phase(qb1, qb2, pi_2)
    }

    /// Add a "tket.qsystem.ZZPhase" op.
    fn add_zz_phase(&mut self, qb1: Wire, qb2: Wire, angle: Wire) -> Result<[Wire; 2], BuildError> {
        Ok(self
            .add_dataflow_op(QSystemOp::ZZPhase, [qb1, qb2, angle])?
            .outputs_arr())
    }

    /// Add a "tket.qsystem.PhasedX" op.
    fn add_phased_x(&mut self, qb: Wire, angle1: Wire, angle2: Wire) -> Result<Wire, BuildError> {
        Ok(self
            .add_dataflow_op(QSystemOp::PhasedX, [qb, angle1, angle2])?
            .out_wire(0))
    }

    /// Add a "tket.qsystem.Rz" op.
    fn add_rz(&mut self, qb: Wire, angle: Wire) -> Result<Wire, BuildError> {
        Ok(self
            .add_dataflow_op(QSystemOp::Rz, [qb, angle])?
            .out_wire(0))
    }

    /// Add a "tket.qsystem.TryQAlloc" op.
    fn add_try_alloc(&mut self) -> Result<Wire, BuildError> {
        Ok(self.add_dataflow_op(QSystemOp::TryQAlloc, [])?.out_wire(0))
    }

    /// Add a "tket.qsystem.QFree" op.
    fn add_qfree(&mut self, qb: Wire) -> Result<(), BuildError> {
        self.add_dataflow_op(QSystemOp::QFree, [qb])?;
        Ok(())
    }

    /// Add a "tket.qsystem.MeasureReset" op.
    /// This operation is equivalent to a `Measure` followed by a `Reset`.
    fn add_measure_reset(&mut self, qb: Wire) -> Result<[Wire; 2], BuildError> {
        Ok(self
            .add_dataflow_op(QSystemOp::MeasureReset, [qb])?
            .outputs_arr())
    }

    /// Add a "tket.qsystem.RuntimeBarrier" op.
    fn add_runtime_barrier(&mut self, qbs: Wire, array_size: u64) -> Result<Wire, BuildError> {
        let op = runtime_barrier_ext_op(array_size)?;
        Ok(self.add_dataflow_op(op, [qbs])?.out_wire(0))
    }

    /// Build a hadamard gate in terms of QSystem primitives.
    fn build_h(&mut self, qb: Wire) -> Result<Wire, BuildError> {
        let pi = pi_mul_f64(self, 1.0);
        let pi_2 = pi_mul_f64(self, 0.5);
        let pi_minus_2 = pi_mul_f64(self, -0.5);

        let q = self.add_phased_x(qb, pi_2, pi_minus_2)?;
        self.add_rz(q, pi)
    }

    /// Build an X gate in terms of QSystem primitives.
    fn build_x(&mut self, qb: Wire) -> Result<Wire, BuildError> {
        let pi = pi_mul_f64(self, 1.0);
        let zero = pi_mul_f64(self, 0.0);
        self.add_phased_x(qb, pi, zero)
    }

    /// Build a Y gate in terms of QSystem primitives.
    fn build_y(&mut self, qb: Wire) -> Result<Wire, BuildError> {
        let pi = pi_mul_f64(self, 1.0);
        let pi_2 = pi_mul_f64(self, 0.5);
        self.add_phased_x(qb, pi, pi_2)
    }

    /// Build a Z gate in terms of QSystem primitives.
    fn build_z(&mut self, qb: Wire) -> Result<Wire, BuildError> {
        let pi = pi_mul_f64(self, 1.0);
        self.add_rz(qb, pi)
    }

    /// Build an S gate in terms of QSystem primitives.
    fn build_s(&mut self, qb: Wire) -> Result<Wire, BuildError> {
        let pi_2 = pi_mul_f64(self, 0.5);
        self.add_rz(qb, pi_2)
    }

    /// Build an Sdg gate in terms of QSystem primitives.
    fn build_sdg(&mut self, qb: Wire) -> Result<Wire, BuildError> {
        let pi_minus_2 = pi_mul_f64(self, -0.5);
        self.add_rz(qb, pi_minus_2)
    }

    /// Build a V gate in terms of QSystem primitives.
    fn build_v(&mut self, qb: Wire) -> Result<Wire, BuildError> {
        let pi_2 = pi_mul_f64(self, 0.5);
        let zero = pi_mul_f64(self, 0.0);
        self.add_phased_x(qb, pi_2, zero)
    }

    /// Build a Vdg gate in terms of QSystem primitives.
    fn build_vdg(&mut self, qb: Wire) -> Result<Wire, BuildError> {
        let pi_minus_2 = pi_mul_f64(self, -0.5);
        let zero = pi_mul_f64(self, 0.0);
        self.add_phased_x(qb, pi_minus_2, zero)
    }

    /// Build a T gate in terms of QSystem primitives.
    fn build_t(&mut self, qb: Wire) -> Result<Wire, BuildError> {
        let pi_4 = pi_mul_f64(self, 0.25);
        self.add_rz(qb, pi_4)
    }

    /// Build a Tdg gate in terms of QSystem primitives.
    fn build_tdg(&mut self, qb: Wire) -> Result<Wire, BuildError> {
        let pi_minus_4 = pi_mul_f64(self, -0.25);
        self.add_rz(qb, pi_minus_4)
    }

    /// Build a CNOT gate in terms of QSystem primitives.
    fn build_cx(&mut self, c: Wire, t: Wire) -> Result<[Wire; 2], BuildError> {
        let pi = pi_mul_f64(self, 1.0);
        let pi_2 = pi_mul_f64(self, 0.5);
        let pi_minus_2 = pi_mul_f64(self, -0.5);

        let t = self.add_phased_x(t, pi_minus_2, pi_2)?;
        let [c, t] = self.build_zz_max(c, t)?;
        let c = self.add_rz(c, pi_minus_2)?;
        let t = self.add_phased_x(t, pi_2, pi)?;
        let t = self.add_rz(t, pi_minus_2)?;

        Ok([c, t])
    }

    /// Build a CY gate in terms of QSystem primitives.
    fn build_cy(&mut self, a: Wire, b: Wire) -> Result<[Wire; 2], BuildError> {
        let pi = pi_mul_f64(self, 1.0);
        let pi_2 = pi_mul_f64(self, 0.5);
        let pi_minus_2 = pi_mul_f64(self, -0.5);

        let a = self.add_phased_x(a, pi, pi)?;
        let b = self.add_phased_x(b, pi_minus_2, pi)?;
        let [a, b] = self.build_zz_max(a, b)?;
        let a = self.add_phased_x(a, pi, pi_2)?;
        let b = self.add_phased_x(b, pi_minus_2, pi_minus_2)?;
        let a = self.add_rz(a, pi_minus_2)?;
        let b = self.add_rz(b, pi_2)?;
        Ok([a, b])
    }

    /// Build a CZ gate in terms of QSystem primitives.
    fn build_cz(&mut self, a: Wire, b: Wire) -> Result<[Wire; 2], BuildError> {
        let pi_2 = pi_mul_f64(self, 0.5);

        let [a, b] = self.build_zz_max(a, b)?;
        let b = self.add_rz(b, pi_2)?;
        let a = self.add_rz(a, pi_2)?;

        Ok([a, b])
    }

    /// Build a RX gate in terms of QSystem primitives.
    fn build_rx(&mut self, qb: Wire, theta: Wire) -> Result<Wire, BuildError> {
        let zero = pi_mul_f64(self, 0.0);
        self.add_phased_x(qb, theta, zero)
    }

    /// Build a RY gate in terms of QSystem primitives.
    fn build_ry(&mut self, qb: Wire, theta: Wire) -> Result<Wire, BuildError> {
        let pi_2 = pi_mul_f64(self, 0.5);
        self.add_phased_x(qb, theta, pi_2)
    }

    /// Build a CRZ gate in terms of QSystem primitives.
    fn build_crz(&mut self, a: Wire, b: Wire, lambda: Wire) -> Result<[Wire; 2], BuildError> {
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

    /// Build a Toffoli (CCX) gate in terms of QSystem primitives.
    fn build_toffoli(&mut self, a: Wire, b: Wire, c: Wire) -> Result<[Wire; 3], BuildError> {
        let pi = pi_mul_f64(self, 1.0);
        let pi_2 = pi_mul_f64(self, 0.5);
        let pi_minus_2 = pi_mul_f64(self, -0.5);
        let pi_4 = pi_mul_f64(self, 0.25);
        let pi_minus_4 = pi_mul_f64(self, -0.25);
        let pi_minus_3_4 = pi_mul_f64(self, -0.75);
        let zero = pi_mul_f64(self, 0.0);

        let c = self.add_phased_x(c, pi, pi_minus_2)?;
        let [b, c] = self.build_zz_max(b, c)?;
        let c = self.add_phased_x(c, pi_4, pi_2)?;
        let [a, c] = self.build_zz_max(a, c)?;
        let c = self.add_phased_x(c, pi_4, zero)?;
        let [b, c] = self.build_zz_max(b, c)?;
        let c = self.add_phased_x(c, pi_4, pi_minus_2)?;
        let [a, c] = self.build_zz_max(a, c)?;
        let a = self.add_phased_x(a, pi, pi_4)?;
        let c = self.add_phased_x(c, pi_minus_3_4, pi)?;
        let [a, b] = self.add_zz_phase(a, b, pi_4)?;
        let c = self.add_rz(c, pi)?;
        let a = self.add_phased_x(a, pi, pi_minus_4)?;
        let b = self.add_rz(b, pi_minus_3_4)?;
        let a = self.add_rz(a, pi_4)?;

        Ok([a, b, c])
    }

    /// Build a projective measurement with a conditional flip.
    fn build_measure_flip(&mut self, qb: Wire) -> Result<[Wire; 2], BuildError> {
        let [qb, b] = self.add_measure_reset(qb)?;
        let sum_b = self.add_dataflow_op(BoolOp::read, [b])?.out_wire(0);
        let mut conditional = self.conditional_builder(
            ([type_row![], type_row![]], sum_b),
            [(qb_t(), qb)],
            vec![qb_t()].into(),
        )?;

        // case 0: 0 state measured, leave alone
        let case0 = conditional.case_builder(0)?;
        let [qb] = case0.input_wires_arr();
        case0.finish_with_outputs([qb])?;

        // case 1: 1 state measured, flip
        let mut case1 = conditional.case_builder(1)?;
        let [qb] = case1.input_wires_arr();
        let qb = case1.build_x(qb)?;
        case1.finish_with_outputs([qb])?;

        let [qb] = conditional.finish_sub_container()?.outputs_arr();
        Ok([qb, sum_b])
    }

    /// Build a qalloc operation that panics on failure.
    fn build_qalloc(&mut self) -> Result<Wire, BuildError> {
        let maybe_qb = self.add_try_alloc()?;
        let [qb] = self.build_expect_sum(1, option_type(qb_t()), maybe_qb, |_| {
            "No more qubits available to allocate.".to_string()
        })?;
        Ok(qb)
    }

    /// Build an array from qubit wires, apply a barrier, and unwrap the array afterwards.
    fn build_wrapped_barrier(
        &mut self,
        qbs: impl IntoIterator<Item = Wire>,
    ) -> Result<Vec<Wire>, BuildError>
    where
        Self: Sized,
    {
        let qbs: Vec<_> = qbs.into_iter().collect();
        let size = qbs.len() as u64;
        let q_arr = self.add_new_array(qb_t(), qbs)?;
        let q_arr = self.add_runtime_barrier(q_arr, size)?;

        self.add_array_unpack(qb_t(), size, q_arr)
    }
}

/// Build a runtime barrier operation on an array of qubits given its size.
pub(crate) fn runtime_barrier_ext_op(
    array_size: u64,
) -> Result<ExtensionOp, hugr::extension::SignatureError> {
    ExtensionOp::new(
        EXTENSION.get_op(&RUNTIME_BARRIER_NAME).unwrap().clone(),
        [TypeArg::BoundedNat(array_size)],
    )
}

impl<D: Dataflow> QSystemOpBuilder for D {}

#[cfg(test)]
mod test {
    use std::sync::Arc;

    use cool_asserts::assert_matches;
    use futures::FutureOpBuilder as _;
    use hugr::builder::{DataflowHugr, FunctionBuilder};
    use hugr::extension::simple_op::MakeExtensionOp;
    use hugr::ops::OpType;
    use hugr::HugrView;
    use strum::IntoEnumIterator as _;

    use super::*;

    fn get_opdef(op: QSystemOp) -> Option<&'static Arc<OpDef>> {
        EXTENSION.get_op(&op.op_id())
    }

    #[test]
    fn create_extension() {
        assert_eq!(EXTENSION.name(), &EXTENSION_ID);

        for o in QSystemOp::iter() {
            assert_eq!(QSystemOp::from_def(get_opdef(o).unwrap()), Ok(o));
        }
    }

    #[test]
    fn lazy_circuit() {
        let hugr = {
            let mut func_builder =
                FunctionBuilder::new("circuit", Signature::new(qb_t(), vec![qb_t(), bool_t()]))
                    .unwrap();
            let [qb] = func_builder.input_wires_arr();
            let [qb, lazy_b] = func_builder.add_lazy_measure_reset(qb).unwrap();
            let [b] = func_builder.add_read(lazy_b, bool_t()).unwrap();
            func_builder.finish_hugr_with_outputs([qb, b]).unwrap()
        };
        assert_matches!(hugr.validate(), Ok(_));
    }

    #[test]
    fn leaked() {
        let hugr = {
            let mut func_builder =
                FunctionBuilder::new("leaked", Signature::new(qb_t(), vec![int_type(6)])).unwrap();
            let [qb] = func_builder.input_wires_arr();
            let lazy_i = func_builder.add_lazy_measure_leaked(qb).unwrap();
            let [i] = func_builder.add_read(lazy_i, int_type(6)).unwrap();
            func_builder.finish_hugr_with_outputs([i]).unwrap()
        };
        assert_matches!(hugr.validate(), Ok(_));
    }

    #[test]
    fn all_ops() {
        let hugr = {
            let mut func_builder = FunctionBuilder::new(
                "all_ops",
                Signature::new(vec![qb_t(), float64_type()], vec![bool_type()]),
            )
            .unwrap();
            let [q0, angle] = func_builder.input_wires_arr();
            let q1 = func_builder.build_qalloc().unwrap();
            let q0 = func_builder.add_reset(q0).unwrap();
            let q1 = func_builder.add_phased_x(q1, angle, angle).unwrap();
            let [q0, q1] = func_builder.build_zz_max(q0, q1).unwrap();
            let [q0, q1] = func_builder.add_zz_phase(q0, q1, angle).unwrap();

            let [q0, q1] = func_builder
                .build_wrapped_barrier([q0, q1])
                .unwrap()
                .try_into()
                .unwrap();

            let q0 = func_builder.add_rz(q0, angle).unwrap();
            let [q0, _b] = func_builder.add_measure_reset(q0).unwrap();
            let b = func_builder.add_measure(q0).unwrap();
            func_builder.add_qfree(q1).unwrap();

            func_builder.finish_hugr_with_outputs([b]).unwrap()
        };
        hugr.validate().unwrap()
    }

    #[test]
    fn test_cast() {
        // test overlapping names don't cause cast errors
        for op in QSystemOp::iter() {
            let optype: OpType = op.into();
            let new_op: QSystemOp = optype.cast().unwrap();
            assert_eq!(op, new_op);
            assert_eq!(optype.cast::<tket::TketOp>(), None);
        }
    }
}
