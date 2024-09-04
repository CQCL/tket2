use hugr::{
    builder::{DFGBuilder, Dataflow, DataflowHugr},
    ops::{self, OpTrait},
    std_extensions::arithmetic::float_types::ConstF64,
    types::Signature,
    Wire,
};
use itertools::Itertools;
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

fn build_angle_rz(
    b: &mut impl Dataflow,
    q: Wire,
    angle: Wire,
) -> Result<hugr::Wire, hugr::builder::BuildError> {
    let float = atorad(b, angle);
    b.add_rz(q, float)
}

/// Lower `Tk2Op` operations to `HSeriesOp` operations.
pub fn lower_tk2_op(
    mut hugr: impl hugr::hugr::hugrmut::HugrMut,
) -> Result<Vec<hugr::Node>, Box<dyn std::error::Error>> {
    let replaced_nodes = hugr::algorithms::replace_many_ops(&mut hugr, |op| {
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
    .collect::<Vec<_>>();

    let lowered_nodes = hugr::algorithms::lower_ops(&mut hugr, |op| {
        let sig = op.dataflow_signature()?;
        let sig = Signature::new(sig.input, sig.output); // ignore extension delta
        let op = op.cast()?;
        let mut b = DFGBuilder::new(sig).ok()?;
        let mut inputs = b.input_wires();
        let outputs = match op {
            Tk2Op::H => vec![b.build_h(inputs.next()?).ok()?],

            Tk2Op::Rz => {
                let [q, angle] = inputs.collect_vec().try_into().ok()?;

                vec![build_angle_rz(&mut b, q, angle).ok()?]
            }
            _ => return None,
        };
        b.finish_hugr_with_outputs(outputs, &REGISTRY).ok()
    })?;

    Ok([replaced_nodes, lowered_nodes].concat())
}

#[cfg(test)]
mod test {
    use hugr::{builder::FunctionBuilder, type_row, HugrView};
    use tket2::Circuit;

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

        let lowered = lower_tk2_op(&mut h).unwrap();
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
    }

    #[rstest]
    #[case(Tk2Op::H, vec![HSeriesOp::PhasedX, HSeriesOp::Rz])]
    #[case(Tk2Op::Rz, vec![HSeriesOp::Rz])]
    fn test_lower(#[case] t2op: Tk2Op, #[case] hseries_ops: Vec<HSeriesOp>) {
        // build dfg with just the op

        use ops::{handle::NodeHandle, OpType};
        let optype: OpType = t2op.into();
        let sig = optype.dataflow_signature().unwrap();
        let mut b = DFGBuilder::new(sig).unwrap();
        let n = b.add_dataflow_op(optype, b.input_wires()).unwrap();
        let mut h = b.finish_hugr_with_outputs(n.outputs(), &REGISTRY).unwrap();

        let lowered = lower_tk2_op(&mut h).unwrap();
        assert_eq!(lowered, vec![n.node()]);
        let circ = Circuit::new(&h, h.root());
        let ops: Vec<HSeriesOp> = circ
            .commands()
            .filter_map(|com| com.optype().cast())
            .collect();
        assert_eq!(ops, hseries_ops);
    }
}
