use hugr::{
    builder::{BuildError, DFGBuilder, Dataflow, DataflowHugr},
    hugr::{hugrmut::HugrMut, HugrError},
    ops::{self, OpTrait},
    std_extensions::arithmetic::float_types::ConstF64,
    types::Signature,
    Hugr, HugrView, Node, Wire,
};
use itertools::Either;
use thiserror::Error;
use tket2::{extension::angle::AngleOp, Tk2Op};

use crate::extension::hseries::{HSeriesOp, HSeriesOpBuilder};

use super::REGISTRY;

pub(super) fn pi_mul(builder: &mut impl Dataflow, multiplier: f64) -> Wire {
    const_f64(builder, multiplier * std::f64::consts::PI)
}

fn const_f64(builder: &mut impl Dataflow, value: f64) -> Wire {
    builder.add_load_const(ops::Const::new(ConstF64::new(value).into()))
}

fn atorad(builder: &mut impl Dataflow, angle: Wire) -> Wire {
    builder
        .add_dataflow_op(AngleOp::atorad, [angle])
        .unwrap()
        .outputs()
        .next()
        .unwrap()
}

/// Errors produced by the [`op_to_hugr`] function.
#[derive(Debug, Error)]
#[error(transparent)]
pub enum LowerBuildError {
    #[error("Error when building the circuit: {0}")]
    BuildError(#[from] BuildError),

    #[error("Unrecognised operation: {0:?} with {1} inputs")]
    UnknownOp(Tk2Op, usize),
}

fn op_to_hugr(op: Tk2Op) -> Result<Hugr, LowerBuildError> {
    let optype: ops::OpType = op.into();
    let sig = optype.dataflow_signature().expect("known to be dataflow");
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
            let float = atorad(&mut b, *angle);
            vec![b.build_rx(*q, float)?]
        }
        (Tk2Op::Ry, [q, angle]) => {
            let float = atorad(&mut b, *angle);
            vec![b.build_ry(*q, float)?]
        }
        (Tk2Op::Rz, [q, angle]) => {
            let float = atorad(&mut b, *angle);
            vec![b.add_rz(*q, float)?]
        }
        (Tk2Op::CRz, [c, t, angle]) => {
            let float = atorad(&mut b, *angle);
            b.build_crz(*c, *t, float)?.into()
        }
        (Tk2Op::Toffoli, [a, b_, c]) => b.build_toffoli(*a, *b_, *c)?.into(),
        (Tk2Op::QAlloc | Tk2Op::QFree | Tk2Op::Reset | Tk2Op::Measure, _) => {
            unreachable!("should be covered by lower_direct")
        }
        _ => return Err(LowerBuildError::UnknownOp(op, inputs.len())), // non-exhaustive
    };
    Ok(b.finish_hugr_with_outputs(outputs, &REGISTRY)?)
}

/// Lower `Tk2Op` operations to `HSeriesOp` operations.
pub fn lower_tk2_op(
    mut hugr: impl HugrMut,
) -> Result<Vec<hugr::Node>, Either<HugrError, hugr::algorithms::lower::LowerError>> {
    let replaced_nodes = lower_direct(&mut hugr).map_err(Either::Left)?;
    let lowered_nodes = hugr::algorithms::lower_ops(&mut hugr, |op| op_to_hugr(op.cast()?).ok())
        .map_err(Either::Right)?;

    Ok([replaced_nodes, lowered_nodes].concat())
}

fn lower_direct(hugr: &mut impl HugrMut) -> Result<Vec<Node>, HugrError> {
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

#[cfg(test)]
mod test {
    use hugr::{builder::FunctionBuilder, type_row, HugrView};
    use tket2::{extension::angle::ANGLE_TYPE, Circuit};

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
        // TODO remaining ops
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
        let mut b = DFGBuilder::new(Signature::new(type_row![ANGLE_TYPE], type_row![])).unwrap();
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
        // dfg, input, output, alloc, phasedx, rz, phasedx, free + 4x(float + load)
        assert_eq!(h.node_count(), 16);
        assert_eq!(check_lowered(&h), Ok(()));
    }
}
