use std::collections::HashMap;

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
use strum::IntoEnumIterator;
use thiserror::Error;
use tket2::{extension::rotation::RotationOpBuilder, Tk2Op};

use crate::extension::hseries::{HSeriesOp, HSeriesOpBuilder};

use super::REGISTRY;

pub(super) fn pi_mul_f64<T: Dataflow + ?Sized>(builder: &mut T, multiplier: f64) -> Wire {
    const_f64(builder, multiplier * std::f64::consts::PI)
}

fn const_f64<T: Dataflow + ?Sized>(builder: &mut T, value: f64) -> Wire {
    builder.add_load_const(ops::Const::new(ConstF64::new(value).into()))
}

/// Errors produced by lowering [Tk2Op]s.
#[derive(Debug, Error)]
#[allow(missing_docs)]
pub enum LowerTk2Error {
    #[error("Error when building the circuit: {0}")]
    BuildError(#[from] BuildError),

    #[error("Unrecognised operation: {0:?} with {1} inputs")]
    UnknownOp(Tk2Op, usize),

    #[error("Error when replacing op: {0}")]
    OpReplacement(#[from] HugrError),

    #[error("Error when lowering ops: {0}")]
    CircuitReplacement(#[from] hugr::algorithms::lower::LowerError),

    #[error("Tk2Ops were not lowered: {0:?}")]
    Unlowered(Vec<Node>),

    #[error(transparent)]
    ValidationError(#[from] ValidatePassError),
}

fn op_to_hugr(op: Tk2Op) -> Result<Hugr, LowerTk2Error> {
    let sig = op.into_extension_op().signature();
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
    Ok(b.finish_hugr_with_outputs(outputs, &REGISTRY)?)
}

fn build_to_radians(b: &mut DFGBuilder<Hugr>, rotation: Wire) -> Result<Wire, BuildError> {
    let turns = b.add_toturns(rotation)?;
    let pi = pi_mul_f64(b, 1.0);
    let float = b.add_dataflow_op(FloatOps::fmul, [turns, pi])?.out_wire(0);
    Ok(float)
}

/// Lower `Tk2Op` operations to `HSeriesOp` operations.
pub fn lower_tk2_op(hugr: &mut impl HugrMut) -> Result<Vec<hugr::Node>, LowerTk2Error> {
    let replaced_nodes = lower_direct(hugr)?;
    let mut hugr_map: HashMap<Tk2Op, Hugr> = HashMap::new();
    for op in Tk2Op::iter() {
        match op_to_hugr(op) {
            Ok(h) => hugr_map.insert(op, h),
            // filter out unknown ops, includes those covered by direct lowering
            Err(LowerTk2Error::UnknownOp(_, _)) => continue,
            Err(e) => return Err(e),
        };
    }

    let lowered_nodes = hugr::algorithms::lower_ops(hugr, |op| hugr_map.get(&op.cast()?).cloned())?;

    Ok([replaced_nodes, lowered_nodes].concat())
}

fn lower_direct(hugr: &mut impl HugrMut) -> Result<Vec<Node>, LowerTk2Error> {
    Ok(hugr::algorithms::replace_many_ops(hugr, |op| {
        let op: Tk2Op = op.cast()?;
        Some(match op {
            Tk2Op::QAlloc => HSeriesOp::QAlloc,
            Tk2Op::QFree => HSeriesOp::QFree,
            Tk2Op::Reset => HSeriesOp::Reset,
            Tk2Op::Measure => HSeriesOp::Measure,
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
                (ext.def().extension() == &tket2::extension::TKET2_EXTENSION_ID).then_some(node)
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
/// equivalent graphs made of [HSeriesOp]s.
///
/// Invokes [lower_tk2_op]. If validation is enabled the resulting HUGR is
/// checked with [check_lowered].
#[derive(Default, Debug, Clone)]
pub struct LowerTket2ToHSeriesPass(ValidationLevel);

impl LowerTket2ToHSeriesPass {
    /// Run `LowerTket2ToHSeriesPass` on the given [HugrMut]. `registry` is used
    /// for validation, if enabled.
    pub fn run(
        &self,
        hugr: &mut impl HugrMut,
        registry: &ExtensionRegistry,
    ) -> Result<(), LowerTk2Error> {
        self.0.run_validated_pass(hugr, registry, |hugr, level| {
            lower_tk2_op(hugr)?;
            if *level != ValidationLevel::None {
                check_lowered(hugr).map_err(LowerTk2Error::Unlowered)?;
            }
            Ok(())
        })
    }

    /// Returns a new `LowerTket2ToHSeriesPass` with the given [ValidationLevel].
    pub fn with_validation_level(&self, level: ValidationLevel) -> Self {
        Self(level)
    }
}

#[cfg(test)]
mod test {
    use hugr::{builder::FunctionBuilder, type_row, HugrView};
    use tket2::{extension::rotation::ROTATION_TYPE, Circuit};

    use super::*;
    use rstest::rstest;

    #[test]
    fn test_lower_direct() {
        let mut b = FunctionBuilder::new("circuit", Signature::new_endo(type_row![])).unwrap();
        let [q] = b.add_dataflow_op(Tk2Op::QAlloc, []).unwrap().outputs_arr();
        let [q] = b.add_dataflow_op(Tk2Op::Reset, [q]).unwrap().outputs_arr();
        let [q, _] = b
            .add_dataflow_op(Tk2Op::Measure, [q])
            .unwrap()
            .outputs_arr();
        b.add_dataflow_op(Tk2Op::QFree, [q]).unwrap();
        let mut h = b.finish_hugr_with_outputs([], &REGISTRY).unwrap();

        let lowered = lower_direct(&mut h).unwrap();
        assert_eq!(lowered.len(), 4);
        let circ = Circuit::new(&h, h.root());
        let ops: Vec<HSeriesOp> = circ
            .commands()
            .map(|com| com.optype().cast().unwrap())
            .collect();
        assert_eq!(
            ops,
            vec![
                HSeriesOp::QAlloc,
                HSeriesOp::Reset,
                HSeriesOp::Measure,
                HSeriesOp::QFree
            ]
        );
        assert_eq!(check_lowered(&h), Ok(()));
    }

    #[rstest]
    #[case(Tk2Op::H, Some(vec![HSeriesOp::PhasedX, HSeriesOp::Rz]))]
    #[case(Tk2Op::X, Some(vec![HSeriesOp::PhasedX]))]
    #[case(Tk2Op::Y, Some(vec![HSeriesOp::PhasedX]))]
    #[case(Tk2Op::Z, Some(vec![HSeriesOp::Rz]))]
    #[case(Tk2Op::S, Some(vec![HSeriesOp::Rz]))]
    #[case(Tk2Op::Sdg, Some(vec![HSeriesOp::Rz]))]
    #[case(Tk2Op::T, Some(vec![HSeriesOp::Rz]))]
    #[case(Tk2Op::Tdg, Some(vec![HSeriesOp::Rz]))]
    #[case(Tk2Op::Rx, Some(vec![HSeriesOp::PhasedX]))]
    #[case(Tk2Op::Ry, Some(vec![HSeriesOp::PhasedX]))]
    #[case(Tk2Op::Rz, Some(vec![HSeriesOp::Rz]))]
    // multi qubit ordering is not deterministic
    #[case(Tk2Op::CX, None)]
    #[case(Tk2Op::CY, None)]
    #[case(Tk2Op::CZ, None)]
    #[case(Tk2Op::CRz, None)]
    #[case(Tk2Op::Toffoli, None)]
    fn test_lower(#[case] t2op: Tk2Op, #[case] hseries_ops: Option<Vec<HSeriesOp>>) {
        // build dfg with just the op

        let h = op_to_hugr(t2op).unwrap();
        let circ = Circuit::new(&h, h.root());
        let ops: Vec<HSeriesOp> = circ
            .commands()
            .filter_map(|com| com.optype().cast())
            .collect();
        if let Some(hseries_ops) = hseries_ops {
            assert_eq!(ops, hseries_ops);
        }

        assert_eq!(check_lowered(&h), Ok(()));
    }

    #[test]
    fn test_mixed() {
        let mut b = DFGBuilder::new(Signature::new(type_row![ROTATION_TYPE], type_row![])).unwrap();
        let [angle] = b.input_wires_arr();
        let [q] = b.add_dataflow_op(Tk2Op::QAlloc, []).unwrap().outputs_arr();
        let [q] = b.add_dataflow_op(Tk2Op::H, [q]).unwrap().outputs_arr();
        let [q] = b
            .add_dataflow_op(Tk2Op::Rx, [q, angle])
            .unwrap()
            .outputs_arr();
        b.add_dataflow_op(Tk2Op::QFree, [q]).unwrap();
        let mut h = b.finish_hugr_with_outputs([], &REGISTRY).unwrap();

        let lowered = lower_tk2_op(&mut h).unwrap();
        assert_eq!(lowered.len(), 4);
        // dfg, input, output, alloc, phasedx, rz, toturns, fmul, phasedx, free + 5x(float + load)
        assert_eq!(h.node_count(), 20);
        assert_eq!(check_lowered(&h), Ok(()));
    }
}
