use derive_more::{Display, Error, From};
use hugr::ops::NamedOp;
use hugr::{
    algorithms::validation::{ValidatePassError, ValidationLevel},
    builder::{BuildError, DFGBuilder, Dataflow, DataflowHugr},
    extension::ExtensionRegistry,
    hugr::{hugrmut::HugrMut, HugrError},
    ops::{self, DataflowOpTrait},
    std_extensions::arithmetic::{float_ops::FloatOps, float_types::ConstF64},
    types::Signature,
    Hugr, HugrView, Node, Wire,
};
use lazy_static::lazy_static;
use std::collections::HashMap;
use strum::IntoEnumIterator;
use tket2::{extension::rotation::RotationOpBuilder, Tk2Op};

use crate::extension::qsystem::{self, QSystemOp, QSystemOpBuilder};

lazy_static! {
    /// Extension registry including [crate::extension::qsystem::REGISTRY] and
    /// [tket2::extension::rotation::ROTATION_EXTENSION].
    pub static ref REGISTRY: ExtensionRegistry = {
        let mut registry = qsystem::REGISTRY.to_owned();
        registry.register(tket2::extension::rotation::ROTATION_EXTENSION.to_owned()).unwrap();
        registry
    };
}

pub(super) fn pi_mul_f64<T: Dataflow + ?Sized>(builder: &mut T, multiplier: f64) -> Wire {
    const_f64(builder, multiplier * std::f64::consts::PI)
}

fn const_f64<T: Dataflow + ?Sized>(builder: &mut T, value: f64) -> Wire {
    builder.add_load_const(ops::Const::new(ConstF64::new(value).into()))
}

/// Errors produced by lowering [Tk2Op]s.
#[derive(Debug, Display, Error, From)]
#[non_exhaustive]
pub enum LowerTk2Error {
    /// An error raised when building the circuit.
    #[display("Error when building the circuit: {_0}")]
    BuildError(BuildError),
    /// Found an unrecognised operation.
    #[display("Unrecognised operation: {} with {_1} inputs", _0.name())]
    UnknownOp(Tk2Op, usize),
    /// An error raised when replacing an operation.
    #[display("Error when replacing op: {_0}")]
    OpReplacement(HugrError),
    /// An error raised when lowering operations.
    #[display("Error when lowering ops: {_0}")]
    CircuitReplacement(hugr::algorithms::lower::LowerError),
    /// Tk2Ops were not lowered after the pass.
    #[display("Tk2Ops were not lowered: {missing_ops:?}")]
    #[from(ignore)]
    Unlowered {
        /// The list of nodes that were not lowered.
        missing_ops: Vec<Node>,
    },
    /// Validation error in the final hugr.
    ValidationError(ValidatePassError),
}

fn op_to_hugr(op: Tk2Op) -> Result<Hugr, LowerTk2Error> {
    let sig = op.into_extension_op().signature().into_owned();
    let sig = Signature::new(sig.input, sig.output); // ignore extension delta
    let mut b = DFGBuilder::new(sig)?;
    let inputs: Vec<_> = b.input_wires().collect();

    let outputs = match (op, inputs.as_slice()) {
        (Tk2Op::H, [q]) => vec![b.build_h(*q)?],
        (Tk2Op::X, [q]) => vec![b.build_x(*q)?],
        (Tk2Op::Y, [q]) => vec![b.build_y(*q)?],
        (Tk2Op::Z, [q]) => vec![b.build_z(*q)?],
        (Tk2Op::S, [q]) => vec![b.build_s(*q)?],
        (Tk2Op::Sdg, [q]) => vec![b.build_sdg(*q)?],
        (Tk2Op::T, [q]) => vec![b.build_t(*q)?],
        (Tk2Op::Tdg, [q]) => vec![b.build_tdg(*q)?],
        (Tk2Op::Measure, [q]) => b.build_measure_flip(*q)?.into(),
        (Tk2Op::QAlloc, []) => vec![b.build_qalloc()?],
        (Tk2Op::CX, [c, t]) => b.build_cx(*c, *t)?.into(),
        (Tk2Op::CY, [c, t]) => b.build_cy(*c, *t)?.into(),
        (Tk2Op::CZ, [c, t]) => b.build_cz(*c, *t)?.into(),
        (Tk2Op::Rx, [q, angle]) => {
            let float = build_to_radians(&mut b, *angle)?;
            vec![b.build_rx(*q, float)?]
        }
        (Tk2Op::Ry, [q, angle]) => {
            let float = build_to_radians(&mut b, *angle)?;
            vec![b.build_ry(*q, float)?]
        }
        (Tk2Op::Rz, [q, angle]) => {
            let float = build_to_radians(&mut b, *angle)?;
            vec![b.add_rz(*q, float)?]
        }
        (Tk2Op::CRz, [c, t, angle]) => {
            let float = build_to_radians(&mut b, *angle)?;
            b.build_crz(*c, *t, float)?.into()
        }
        (Tk2Op::Toffoli, [a, b_, c]) => b.build_toffoli(*a, *b_, *c)?.into(),
        _ => return Err(LowerTk2Error::UnknownOp(op, inputs.len())), // non-exhaustive
    };
    Ok(b.finish_hugr_with_outputs(outputs)?)
}

fn build_to_radians(b: &mut DFGBuilder<Hugr>, rotation: Wire) -> Result<Wire, BuildError> {
    let turns = b.add_to_halfturns(rotation)?;
    let pi = pi_mul_f64(b, 1.0);
    let float = b.add_dataflow_op(FloatOps::fmul, [turns, pi])?.out_wire(0);
    Ok(float)
}

/// Lower `Tk2Op` operations to `QSystemOp` operations.
pub fn lower_tk2_op(hugr: &mut impl HugrMut) -> Result<Vec<hugr::Node>, LowerTk2Error> {
    let mut replaced_nodes = lower_direct(hugr)?;
    let mut hugr_map: HashMap<Tk2Op, Hugr> = HashMap::new();
    for op in Tk2Op::iter() {
        match op_to_hugr(op) {
            Ok(h) => hugr_map.insert(op, h),
            // filter out unknown ops, includes those covered by direct lowering
            Err(LowerTk2Error::UnknownOp(_, _)) => continue,
            Err(e) => return Err(e),
        };
    }

    let lowered_nodes = hugr::algorithms::lower_ops(hugr, |op| hugr_map.get(&op.cast()?).cloned())?
        .into_iter()
        .map(|(n, _)| n);

    replaced_nodes.extend(lowered_nodes);
    Ok(replaced_nodes)
}

fn lower_direct(hugr: &mut impl HugrMut) -> Result<Vec<Node>, LowerTk2Error> {
    Ok(hugr::algorithms::replace_many_ops(hugr, |op| {
        let op: Tk2Op = op.cast()?;
        Some(match op {
            Tk2Op::TryQAlloc => QSystemOp::TryQAlloc,
            Tk2Op::QFree => QSystemOp::QFree,
            Tk2Op::Reset => QSystemOp::Reset,
            Tk2Op::MeasureFree => QSystemOp::Measure,
            _ => return None,
        })
    })?
    .into_iter()
    .map(|(node, _)| node)
    .collect())
}

/// Check there are no "tket2.quantum" ops left in the HUGR.
///
/// # Errors
/// Returns vector of nodes that are not lowered.
pub fn check_lowered(hugr: &impl HugrView) -> Result<(), Vec<Node>> {
    let unlowered: Vec<Node> = hugr
        .nodes()
        .filter_map(|node| {
            let optype = hugr.get_optype(node);
            optype.as_extension_op().and_then(|ext| {
                (ext.def().extension_id() == &tket2::extension::TKET2_EXTENSION_ID).then_some(node)
            })
        })
        .collect();

    if unlowered.is_empty() {
        Ok(())
    } else {
        Err(unlowered)
    }
}

/// A `Hugr -> Hugr` pass that replaces [tket2::Tk2Op] nodes to
/// equivalent graphs made of [QSystemOp]s.
///
/// Invokes [lower_tk2_op]. If validation is enabled the resulting HUGR is
/// checked with [check_lowered].
#[derive(Default, Debug, Clone)]
pub struct LowerTket2ToQSystemPass(ValidationLevel);

impl LowerTket2ToQSystemPass {
    /// Run `LowerTket2ToQSystemPass` on the given [HugrMut]. `registry` is used
    /// for validation, if enabled.
    pub fn run(&self, hugr: &mut impl HugrMut) -> Result<(), LowerTk2Error> {
        self.0.run_validated_pass(hugr, |hugr, level| {
            lower_tk2_op(hugr)?;
            if *level != ValidationLevel::None {
                check_lowered(hugr)
                    .map_err(|missing_ops| LowerTk2Error::Unlowered { missing_ops })?;
            }
            Ok(())
        })
    }

    /// Returns a new `LowerTket2ToQSystemPass` with the given [ValidationLevel].
    pub fn with_validation_level(&self, level: ValidationLevel) -> Self {
        Self(level)
    }
}

#[cfg(test)]
mod test {
    use hugr::{
        builder::FunctionBuilder,
        extension::prelude::{bool_t, option_type, qb_t, UnwrapBuilder as _},
        type_row, HugrView,
    };
    use tket2::{extension::rotation::rotation_type, Circuit};

    use super::*;
    use rstest::rstest;

    #[test]
    fn test_lower_direct() {
        let mut b = FunctionBuilder::new("circuit", Signature::new_endo(type_row![])).unwrap();
        let [maybe_q] = b
            .add_dataflow_op(Tk2Op::TryQAlloc, [])
            .unwrap()
            .outputs_arr();
        let [q] = b
            .build_unwrap_sum(&REGISTRY, 1, option_type(qb_t()), maybe_q)
            .unwrap();
        let [q] = b.add_dataflow_op(Tk2Op::Reset, [q]).unwrap().outputs_arr();
        b.add_dataflow_op(Tk2Op::QFree, [q]).unwrap();
        let [maybe_q] = b
            .add_dataflow_op(Tk2Op::TryQAlloc, [])
            .unwrap()
            .outputs_arr();
        let [q] = b
            .build_unwrap_sum(&REGISTRY, 1, option_type(qb_t()), maybe_q)
            .unwrap();

        let [_] = b
            .add_dataflow_op(Tk2Op::MeasureFree, [q])
            .unwrap()
            .outputs_arr();
        let mut h = b
            .finish_hugr_with_outputs([])
            .unwrap_or_else(|e| panic!("{}", e));

        let lowered = lower_direct(&mut h).unwrap();
        assert_eq!(lowered.len(), 5);
        let circ = Circuit::new(&h, h.root());
        let ops: Vec<QSystemOp> = circ
            .commands()
            .filter_map(|com| com.optype().cast())
            .collect();
        assert_eq!(
            ops,
            vec![
                QSystemOp::TryQAlloc,
                QSystemOp::Measure,
                QSystemOp::TryQAlloc,
                QSystemOp::Reset,
                QSystemOp::QFree,
            ]
        );
        assert_eq!(check_lowered(&h), Ok(()));
    }

    #[rstest]
    #[case(Tk2Op::H, Some(vec![QSystemOp::PhasedX, QSystemOp::Rz]))]
    #[case(Tk2Op::X, Some(vec![QSystemOp::PhasedX]))]
    #[case(Tk2Op::Y, Some(vec![QSystemOp::PhasedX]))]
    #[case(Tk2Op::Z, Some(vec![QSystemOp::Rz]))]
    #[case(Tk2Op::S, Some(vec![QSystemOp::Rz]))]
    #[case(Tk2Op::Sdg, Some(vec![QSystemOp::Rz]))]
    #[case(Tk2Op::T, Some(vec![QSystemOp::Rz]))]
    #[case(Tk2Op::Tdg, Some(vec![QSystemOp::Rz]))]
    #[case(Tk2Op::Rx, Some(vec![QSystemOp::PhasedX]))]
    #[case(Tk2Op::Ry, Some(vec![QSystemOp::PhasedX]))]
    #[case(Tk2Op::Rz, Some(vec![QSystemOp::Rz]))]
    // multi qubit ordering is not deterministic
    #[case(Tk2Op::CX, None)]
    #[case(Tk2Op::CY, None)]
    #[case(Tk2Op::CZ, None)]
    #[case(Tk2Op::CRz, None)]
    #[case(Tk2Op::Toffoli, None)]
    // conditional doesn't fit in to commands
    #[case(Tk2Op::Measure, None)]
    #[case(Tk2Op::QAlloc, None)]
    fn test_lower(#[case] t2op: Tk2Op, #[case] qsystem_ops: Option<Vec<QSystemOp>>) {
        // build dfg with just the op

        let h = op_to_hugr(t2op).unwrap();
        let circ = Circuit::new(&h, h.root());
        let ops: Vec<QSystemOp> = circ
            .commands()
            .filter_map(|com| com.optype().cast())
            .collect();
        if let Some(qsystem_ops) = qsystem_ops {
            assert_eq!(ops, qsystem_ops);
        }

        assert_eq!(check_lowered(&h), Ok(()));
    }

    #[test]
    fn test_mixed() {
        let mut b = DFGBuilder::new(Signature::new(rotation_type(), bool_t())).unwrap();
        let [angle] = b.input_wires_arr();
        let [q] = b.add_dataflow_op(Tk2Op::QAlloc, []).unwrap().outputs_arr();
        let [q] = b.add_dataflow_op(Tk2Op::H, [q]).unwrap().outputs_arr();
        let [q] = b
            .add_dataflow_op(Tk2Op::Rx, [q, angle])
            .unwrap()
            .outputs_arr();
        let [q, bool] = b
            .add_dataflow_op(Tk2Op::Measure, [q])
            .unwrap()
            .outputs_arr();
        b.add_dataflow_op(Tk2Op::QFree, [q]).unwrap();
        let mut h = b.finish_hugr_with_outputs([bool]).unwrap();

        let lowered = lower_tk2_op(&mut h).unwrap();
        assert_eq!(lowered.len(), 5);
        // dfg, input, output, alloc + (10 for unwrap), phasedx, rz, toturns, fmul, phasedx, free +
        // 5x(float + load), measure_reset, conditional, case(input, output) * 2, flip
        // (phasedx + 2*(float + load))
        assert_eq!(h.node_count(), 43);
        assert_eq!(check_lowered(&h), Ok(()));
    }
}
