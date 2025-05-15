//! Provides a `ReplaceBoolPass` which replaces the tket2.bool type and
//! lazifies measure operations.
use derive_more::{Display, Error, From};
use hugr::{
    algorithms::{
        non_local::NonLocalEdgesError,
        replace_types::{NodeTemplate, ReplaceTypesError},
        ComposablePass, ReplaceTypes,
    },
    builder::{
        inout_sig, BuildHandle, DFGBuilder, Dataflow, DataflowHugr, DataflowSubContainer,
        SubContainer,
    },
    extension::prelude::{bool_t, qb_t},
    hugr::hugrmut::HugrMut,
    ops::{handle::ConditionalID, ExtensionOp, Tag, Value},
    std_extensions::logic::LogicOp,
    types::{SumType, Type, TypeArg},
    Hugr, Node, Wire,
};
use tket2::extension::{
    bool::{bool_type, ConstBool, BOOL_EXTENSION},
    TKET2_EXTENSION,
};

use crate::extension::{
    futures::{self, future_type, FutureOpBuilder},
    qsystem,
};

#[derive(Error, Debug, Display, From)]
#[non_exhaustive]
/// An error reported from [ReplaceBoolPass].
pub enum ReplaceBoolPassError<N> {
    /// The HUGR was found to contain non-local edges.
    NonLocalEdgesError(NonLocalEdgesError<N>),
    /// There was an error while replacing the type.
    ReplacementError(ReplaceTypesError),
}

/// A HUGR -> HUGR pass which replaces the `tket2.bool`, enabling lazifying of measure
/// operations.
///
/// The `tket2.bool` type is replaced by a sum type of `bool_t` (the standard
/// HUGR bool type represented by a unit sum) and `future(bool_t)`, with its operations
/// being turned into conditionals that read the future if necessary.
///
/// [Tk2Op::Measure], [QSystemOp::Measure], and [QSystemOp::MeasureReset] nodes
/// are replaced by [QSystemOp::LazyMeasure] and [QSystemOp::LazyMeasureReset]
/// nodes.
#[derive(Default, Debug, Clone)]
pub struct ReplaceBoolPass;

impl<H: HugrMut<Node = Node>> ComposablePass<H> for ReplaceBoolPass {
    type Error = ReplaceBoolPassError<H::Node>;
    type Result = ();

    fn run(&self, hugr: &mut H) -> Result<(), Self::Error> {
        // TODO uncomment once https://github.com/CQCL/hugr/issues/1234 is complete
        // ensure_no_nonlocal_edges(hugr)?;
        let lowerer = lowerer();
        lowerer.run(hugr)?;
        Ok(())
    }
}

/// The type each tket2.bool is replaced with.
fn bool_dest() -> Type {
    SumType::new([bool_t(), future_type(bool_t())]).into()
}

fn read_builder(dfb: &mut DFGBuilder<Hugr>, sum_wire: Wire) -> BuildHandle<ConditionalID> {
    let mut cb = dfb
        .conditional_builder(
            ([bool_t().into(), future_type(bool_t()).into()], sum_wire),
            [],
            vec![bool_t()].into(),
        )
        .unwrap();

    // If the input is a bool, we can just return it.
    let case0 = cb.case_builder(0).unwrap();
    let [b] = case0.input_wires_arr();
    case0.finish_with_outputs([b]).unwrap();
    // If the input is a future, we need to read it.
    let mut case1 = cb.case_builder(1).unwrap();
    let [f] = case1.input_wires_arr();
    let [f] = case1.add_read(f, bool_t()).unwrap();
    case1.finish_with_outputs([f]).unwrap();

    cb.finish_sub_container().unwrap()
}

fn read_op_dest() -> NodeTemplate {
    let mut dfb = DFGBuilder::new(inout_sig(vec![bool_dest()], vec![bool_t()])).unwrap();
    let [sum_wire] = dfb.input_wires_arr();
    let cond = read_builder(&mut dfb, sum_wire);
    let h = dfb.finish_hugr_with_outputs(cond.outputs()).unwrap();
    NodeTemplate::CompoundOp(Box::new(h))
}

fn make_opaque_op_dest() -> NodeTemplate {
    let mut dfb = DFGBuilder::new(inout_sig(vec![bool_t()], vec![bool_dest()])).unwrap();
    let [inp] = dfb.input_wires_arr();
    let out = dfb
        .add_dataflow_op(
            Tag::new(0, vec![bool_t().into(), future_type(bool_t()).into()]),
            vec![inp],
        )
        .unwrap();
    let h = dfb.finish_hugr_with_outputs(out.outputs()).unwrap();
    NodeTemplate::CompoundOp(Box::new(h))
}

fn binary_logic_op_dest(op_name: &str) -> NodeTemplate {
    let mut dfb =
        DFGBuilder::new(inout_sig(vec![bool_dest(), bool_dest()], vec![bool_dest()])).unwrap();
    let [sum_wire1, sum_wire2] = dfb.input_wires_arr();
    let cond1 = read_builder(&mut dfb, sum_wire1);
    let cond2 = read_builder(&mut dfb, sum_wire2);
    let result = match op_name {
        "eq" => dfb
            .add_dataflow_op(LogicOp::Eq, [cond1.out_wire(0), cond2.out_wire(0)])
            .unwrap(),
        "and" => dfb
            .add_dataflow_op(LogicOp::And, [cond1.out_wire(0), cond2.out_wire(0)])
            .unwrap(),
        "or" => dfb
            .add_dataflow_op(LogicOp::Or, [cond1.out_wire(0), cond2.out_wire(0)])
            .unwrap(),
        "xor" => dfb
            .add_dataflow_op(LogicOp::Xor, [cond1.out_wire(0), cond2.out_wire(0)])
            .unwrap(),
        _ => panic!("Unknown op name"),
    };
    let out = dfb
        .add_dataflow_op(
            Tag::new(0, vec![bool_t().into(), future_type(bool_t()).into()]),
            vec![result.out_wire(0)],
        )
        .unwrap();

    let h = dfb.finish_hugr_with_outputs(out.outputs()).unwrap();
    NodeTemplate::CompoundOp(Box::new(h))
}

fn not_op_dest() -> NodeTemplate {
    let mut dfb = DFGBuilder::new(inout_sig(vec![bool_dest()], vec![bool_dest()])).unwrap();
    let [sum_wire] = dfb.input_wires_arr();
    let cond = read_builder(&mut dfb, sum_wire);
    let result = dfb
        .add_dataflow_op(LogicOp::Not, [cond.out_wire(0)])
        .unwrap();
    let out = dfb
        .add_dataflow_op(
            Tag::new(0, vec![bool_t().into(), future_type(bool_t()).into()]),
            vec![result.out_wire(0)],
        )
        .unwrap();
    let h = dfb.finish_hugr_with_outputs(out.outputs()).unwrap();
    NodeTemplate::CompoundOp(Box::new(h))
}

fn measure_dest() -> NodeTemplate {
    let lazy_measure = ExtensionOp::new(
        qsystem::EXTENSION.get_op("LazyMeasure").unwrap().clone(),
        vec![],
    )
    .unwrap();

    let mut dfb = DFGBuilder::new(inout_sig(vec![qb_t()], vec![bool_dest()])).unwrap();
    let [q] = dfb.input_wires_arr();
    let measure = dfb.add_dataflow_op(lazy_measure, vec![q]).unwrap();
    let tagged_output = dfb
        .add_dataflow_op(
            Tag::new(1, vec![bool_t().into(), future_type(bool_t()).into()]),
            vec![measure.out_wire(0)],
        )
        .unwrap();
    let h = dfb
        .finish_hugr_with_outputs(tagged_output.outputs())
        .unwrap();
    NodeTemplate::CompoundOp(Box::new(h))
}

fn measure_reset_dest() -> NodeTemplate {
    let lazy_measure_reset = ExtensionOp::new(
        qsystem::EXTENSION
            .get_op("LazyMeasureReset")
            .unwrap()
            .clone(),
        vec![],
    )
    .unwrap();

    let mut dfb = DFGBuilder::new(inout_sig(vec![qb_t()], vec![qb_t(), bool_dest()])).unwrap();
    let [q] = dfb.input_wires_arr();
    let measure = dfb.add_dataflow_op(lazy_measure_reset, vec![q]).unwrap();
    let tagged_output = dfb
        .add_dataflow_op(
            Tag::new(1, vec![bool_t().into(), future_type(bool_t()).into()]),
            vec![measure.out_wire(1)],
        )
        .unwrap();
    let h = dfb
        .finish_hugr_with_outputs(vec![measure.out_wire(0), tagged_output.out_wire(0)])
        .unwrap();
    NodeTemplate::CompoundOp(Box::new(h))
}

/// The configuration used for replacing tket2.bool extension types and ops.
fn lowerer() -> ReplaceTypes {
    let mut lw = ReplaceTypes::default();

    // Replace tket2.bool type.
    lw.replace_type(bool_type().as_extension().unwrap().clone(), bool_dest());
    let bool_arg: TypeArg = bool_t().clone().into();
    let dup_op = ExtensionOp::new(
        futures::EXTENSION.get_op("Dup").unwrap().clone(),
        [bool_arg.clone()],
    )
    .unwrap();
    let free_op = ExtensionOp::new(
        futures::EXTENSION.get_op("Free").unwrap().clone(),
        [bool_arg.clone()],
    )
    .unwrap();
    lw.linearizer()
        .register_simple(
            future_type(bool_t()).as_extension().unwrap().clone(),
            NodeTemplate::SingleOp(dup_op.into()),
            NodeTemplate::SingleOp(free_op.into()),
        )
        .unwrap();

    // Replace tket2.bool constants.
    lw.replace_consts(
        bool_type().as_extension().unwrap().clone(),
        |const_bool, _| {
            Ok(Value::sum(
                0,
                [Value::from_bool(
                    const_bool
                        .value()
                        .downcast_ref::<ConstBool>()
                        .unwrap()
                        .value(),
                )],
                SumType::new([vec![bool_t()], vec![future_type(bool_t())]]),
            )
            .unwrap())
        },
    );

    // Replace all tket2.bool ops.
    let read_op = ExtensionOp::new(BOOL_EXTENSION.get_op("read").unwrap().clone(), vec![]).unwrap();
    lw.replace_op(&read_op, read_op_dest());
    let make_opaque_op = ExtensionOp::new(
        BOOL_EXTENSION.get_op("make_opaque").unwrap().clone(),
        vec![],
    )
    .unwrap();
    lw.replace_op(&make_opaque_op, make_opaque_op_dest());
    for op_name in ["eq", "and", "or", "xor"] {
        let op = ExtensionOp::new(BOOL_EXTENSION.get_op(op_name).unwrap().clone(), vec![]).unwrap();
        lw.replace_op(&op, binary_logic_op_dest(op_name));
    }
    let not_op = ExtensionOp::new(BOOL_EXTENSION.get_op("not").unwrap().clone(), vec![]).unwrap();
    lw.replace_op(&not_op, not_op_dest());

    // Replace measure ops with lazy versions.
    let tket2_measure_free = ExtensionOp::new(
        TKET2_EXTENSION.get_op("MeasureFree").unwrap().clone(),
        vec![],
    )
    .unwrap();
    let qsystem_measure = ExtensionOp::new(
        qsystem::EXTENSION.get_op("Measure").unwrap().clone(),
        vec![],
    )
    .unwrap();
    let qsystem_measure_reset = ExtensionOp::new(
        qsystem::EXTENSION.get_op("MeasureReset").unwrap().clone(),
        vec![],
    )
    .unwrap();
    lw.replace_op(&tket2_measure_free, measure_dest());
    lw.replace_op(&qsystem_measure, measure_dest());
    lw.replace_op(&qsystem_measure_reset, measure_reset_dest());

    lw
}

#[cfg(test)]
mod test {
    use crate::extension::qsystem::{QSystemOp, QSystemOpBuilder};

    use super::*;
    use hugr::ops::OpType;
    use hugr::{
        builder::{inout_sig, DFGBuilder, Dataflow, DataflowHugr},
        extension::prelude::qb_t,
        types::TypeRow,
        HugrView,
    };
    use rstest::rstest;
    use tket2::{
        extension::bool::{BoolOp, BoolOpBuilder},
        Tk2Op,
    };

    fn tket2_bool_t() -> Type {
        bool_type().clone()
    }

    #[test]
    fn test_consts() {
        let mut dfb = DFGBuilder::new(inout_sig(vec![], vec![tket2_bool_t()])).unwrap();
        let const_wire = dfb.add_load_value(ConstBool::new(true));
        let mut h = dfb.finish_hugr_with_outputs([const_wire]).unwrap();

        h.validate().unwrap();
        let pass = ReplaceBoolPass;
        pass.run(&mut h).unwrap();
        h.validate().unwrap();
        let sig = h.signature(h.entrypoint()).unwrap();
        assert_eq!(sig.output(), &TypeRow::from(vec![bool_dest()]));
    }

    #[test]
    fn test_read() {
        let mut dfb = DFGBuilder::new(inout_sig(vec![tket2_bool_t()], vec![bool_t()])).unwrap();
        let [b] = dfb.input_wires_arr();
        let output = dfb.add_bool_read(b).unwrap();
        let mut h = dfb.finish_hugr_with_outputs(output).unwrap();

        assert_eq!(h.num_nodes(), 8);

        let pass = ReplaceBoolPass;
        pass.run(&mut h).unwrap();

        let sig = h.signature(h.entrypoint()).unwrap();
        assert_eq!(sig.input(), &TypeRow::from(vec![bool_dest()]));
        assert_eq!(sig.output(), &TypeRow::from(vec![bool_t()]));

        assert_eq!(h.num_nodes(), 18);
    }

    #[test]
    fn test_make_opaque() {
        let mut dfb = DFGBuilder::new(inout_sig(vec![bool_t()], vec![tket2_bool_t()])).unwrap();
        let [b] = dfb.input_wires_arr();
        let output = dfb.add_bool_make_opaque(b).unwrap();
        let mut h = dfb.finish_hugr_with_outputs(output).unwrap();

        assert_eq!(h.num_nodes(), 8);

        let pass = ReplaceBoolPass;
        pass.run(&mut h).unwrap();

        let sig = h.signature(h.entrypoint()).unwrap();
        assert_eq!(sig.input(), &TypeRow::from(vec![bool_t()]));
        assert_eq!(sig.output(), &TypeRow::from(vec![bool_dest()]));

        assert_eq!(h.num_nodes(), 11);
    }

    #[rstest]
    #[case(BoolOp::eq)]
    #[case(BoolOp::and)]
    #[case(BoolOp::or)]
    #[case(BoolOp::xor)]
    fn test_logic(#[case] logic_op: BoolOp) {
        let mut dfb = DFGBuilder::new(inout_sig(
            vec![tket2_bool_t(), tket2_bool_t()],
            vec![tket2_bool_t()],
        ))
        .unwrap();
        let [b1, b2] = dfb.input_wires_arr();
        let result = dfb.add_dataflow_op(logic_op, [b1, b2]).unwrap();
        let mut h = dfb.finish_hugr_with_outputs(result.outputs()).unwrap();

        let pass = ReplaceBoolPass;
        pass.run(&mut h).unwrap();

        let sig = h.signature(h.entrypoint()).unwrap();
        assert_eq!(sig.input(), &TypeRow::from(vec![bool_dest(), bool_dest()]));
        assert_eq!(sig.output(), &TypeRow::from(vec![bool_dest()]));
    }

    #[test]
    fn test_not() {
        let mut dfb =
            DFGBuilder::new(inout_sig(vec![tket2_bool_t()], vec![tket2_bool_t()])).unwrap();
        let [b] = dfb.input_wires_arr();
        let result = dfb.add_dataflow_op(BoolOp::not, [b]).unwrap();
        let mut h = dfb.finish_hugr_with_outputs(result.outputs()).unwrap();

        let pass = ReplaceBoolPass;
        pass.run(&mut h).unwrap();

        let sig = h.signature(h.entrypoint()).unwrap();
        assert_eq!(sig.input(), &TypeRow::from(vec![bool_dest()]));
        assert_eq!(sig.output(), &TypeRow::from(vec![bool_dest()]));
    }

    #[rstest]
    #[case(Tk2Op::MeasureFree)]
    #[case(QSystemOp::Measure)]
    fn test_measure<T: Into<OpType>>(#[case] measure_op: T) {
        let mut dfb = DFGBuilder::new(inout_sig(vec![qb_t()], vec![bool_type()])).unwrap();
        let [q] = dfb.input_wires_arr();
        let output = dfb.add_dataflow_op(measure_op, [q]).unwrap();
        let mut h = dfb.finish_hugr_with_outputs(output.outputs()).unwrap();

        let pass = ReplaceBoolPass;
        pass.run(&mut h).unwrap();

        let sig = h.signature(h.entrypoint()).unwrap();
        assert_eq!(sig.output(), &TypeRow::from(vec![bool_dest()]));

        // TODO: Figure out how to do this when the op is inside a DFB child.
        // let top_ops = h.children(h.entrypoint()).map(|n| h.get_optype(n)).collect_vec();
        // assert!(top_ops.iter().any(|op| {
        //    if let Some(ext_op) = op.as_extension_op() {
        //        ext_op.def().name() == "LazyMeasure"
        //    } else {
        //        false
        //    }
        //}));
    }

    #[test]
    fn test_measure_reset() {
        let mut dfb = DFGBuilder::new(inout_sig(vec![qb_t()], vec![qb_t(), bool_type()])).unwrap();
        let [q] = dfb.input_wires_arr();
        let output = dfb.add_measure_reset(q).unwrap();
        let mut h = dfb.finish_hugr_with_outputs(output).unwrap();

        let pass = ReplaceBoolPass;
        pass.run(&mut h).unwrap();

        let sig = h.signature(h.entrypoint()).unwrap();
        assert_eq!(sig.output(), &TypeRow::from(vec![qb_t(), bool_dest()]));
    }
}
