//! Provides a `ReplaceBoolPass` which replaces the tket.bool type and
//! lazifies measure operations.
mod static_array;

use derive_more::{Display, Error, From};
use hugr::{
    algorithms::{
        ensure_no_nonlocal_edges,
        non_local::FindNonLocalEdgesError,
        replace_types::{NodeTemplate, ReplaceTypesError, ReplacementOptions},
        ComposablePass, ReplaceTypes,
    },
    builder::{
        inout_sig, BuildHandle, Container, DFGBuilder, Dataflow, DataflowHugr,
        DataflowSubContainer, SubContainer,
    },
    extension::{
        prelude::{bool_t, qb_t},
        simple_op::MakeRegisteredOp,
    },
    hugr::hugrmut::HugrMut,
    ops::{handle::ConditionalID, ExtensionOp, Tag, Value},
    std_extensions::{
        collections::{
            array::{self, array_type, ARRAY_CLONE_OP_ID, ARRAY_DISCARD_OP_ID},
            borrow_array::{self, borrow_array_type},
        },
        logic::LogicOp,
    },
    types::{SumType, Type},
    Hugr, HugrView, Node, Wire,
};
use static_array::{ReplaceStaticArrayBoolPass, ReplaceStaticArrayBoolPassError};
use tket::{
    extension::{
        bool::{bool_type, BoolOp, ConstBool},
        guppy::{DROP_OP_NAME, GUPPY_EXTENSION},
    },
    TketOp,
};

use crate::extension::{
    futures::{future_type, FutureOp, FutureOpBuilder, FutureOpDef},
    qsystem::QSystemOp,
};

#[derive(Error, Debug, Display, From)]
#[non_exhaustive]
/// An error reported from [ReplaceBoolPass].
pub enum ReplaceBoolPassError<N> {
    /// The HUGR was found to contain non-local edges.
    NonLocalEdgesError(FindNonLocalEdgesError<N>),
    /// There was an error while replacing the type.
    ReplacementError(ReplaceTypesError),
    /// There was an error while transforming static arrays containing tket.bool
    /// to static arrays of bool_t.
    ReplaceStaticArrayBoolPassError(ReplaceStaticArrayBoolPassError),
}

/// A HUGR -> HUGR pass which replaces the `tket.bool`, enabling lazifying of measure
/// operations.
///
/// The `tket.bool` type is replaced by a sum type of `bool_t` (the standard
/// HUGR bool type represented by a unit sum) and `future(bool_t)`, with its operations
/// being turned into conditionals that read the future if necessary.
///
/// [TketOp::Measure], [QSystemOp::Measure], and [QSystemOp::MeasureReset] nodes
/// are replaced by [QSystemOp::LazyMeasure] and [QSystemOp::LazyMeasureReset]
/// nodes.
///
/// [TketOp::Measure]: tket::TketOp::Measure
/// [QSystemOp::Measure]: crate::extension::qsystem::QSystemOp::Measure
/// [QSystemOp::MeasureReset]: crate::extension::qsystem::QSystemOp::MeasureReset
/// [QSystemOp::LazyMeasure]: crate::extension::qsystem::QSystemOp::LazyMeasure
/// [QSystemOp::LazyMeasureReset]: crate::extension::qsystem::QSystemOp::LazyMeasureReset
#[derive(Default, Debug, Clone)]
pub struct ReplaceBoolPass;

impl<H: HugrMut<Node = Node>> ComposablePass<H> for ReplaceBoolPass {
    type Error = ReplaceBoolPassError<H::Node>;
    type Result = ();

    fn run(&self, hugr: &mut H) -> Result<(), Self::Error> {
        ensure_no_nonlocal_edges(hugr)?;
        ReplaceStaticArrayBoolPass::default().run(hugr)?;
        let lowerer = lowerer();
        lowerer.run(hugr)?;
        Ok(())
    }
}

/// The type each tket.bool is replaced with.
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

fn binary_logic_op_dest(op: &BoolOp) -> NodeTemplate {
    let mut dfb =
        DFGBuilder::new(inout_sig(vec![bool_dest(), bool_dest()], vec![bool_dest()])).unwrap();
    let [sum_wire1, sum_wire2] = dfb.input_wires_arr();
    let cond1 = read_builder(&mut dfb, sum_wire1);
    let cond2 = read_builder(&mut dfb, sum_wire2);
    let result = match op {
        BoolOp::eq => dfb
            .add_dataflow_op(LogicOp::Eq, [cond1.out_wire(0), cond2.out_wire(0)])
            .unwrap(),
        BoolOp::and => dfb
            .add_dataflow_op(LogicOp::And, [cond1.out_wire(0), cond2.out_wire(0)])
            .unwrap(),
        BoolOp::or => dfb
            .add_dataflow_op(LogicOp::Or, [cond1.out_wire(0), cond2.out_wire(0)])
            .unwrap(),
        BoolOp::xor => dfb
            .add_dataflow_op(LogicOp::Xor, [cond1.out_wire(0), cond2.out_wire(0)])
            .unwrap(),
        op => panic!("Unknown op name: {op:?}"),
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
    let lazy_measure = QSystemOp::LazyMeasure.to_extension_op().unwrap();

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
    let lazy_measure_reset = QSystemOp::LazyMeasureReset.to_extension_op().unwrap();

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

fn array_clone_dest(array_ty: Type) -> NodeTemplate {
    let mut dfb = DFGBuilder::new(inout_sig(
        vec![array_ty.clone()],
        vec![array_ty.clone(), array_ty],
    ))
    .unwrap();
    let mut h = std::mem::take(dfb.hugr_mut());
    let [inp, outp] = h.get_io(h.entrypoint()).unwrap();
    h.connect(inp, 0, outp, 0);
    h.connect(inp, 0, outp, 1);
    NodeTemplate::CompoundOp(Box::new(h))
}

/// The configuration used for replacing tket.bool extension types and ops.
fn lowerer() -> ReplaceTypes {
    let mut lw = ReplaceTypes::default();

    // Replace tket.bool type.
    lw.replace_type(bool_type().as_extension().unwrap().clone(), bool_dest());
    let dup_op = FutureOp {
        op: FutureOpDef::Dup,
        typ: bool_t(),
    }
    .to_extension_op()
    .unwrap();
    let free_op = FutureOp {
        op: FutureOpDef::Free,
        typ: bool_t(),
    }
    .to_extension_op()
    .unwrap();
    lw.linearizer()
        .register_simple(
            future_type(bool_t()).as_extension().unwrap().clone(),
            NodeTemplate::SingleOp(dup_op.into()),
            NodeTemplate::SingleOp(free_op.into()),
        )
        .unwrap();

    // Replace tket.bool constants.
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

    // Replace all tket.bool ops.
    let read_op = BoolOp::read.to_extension_op().unwrap();
    lw.replace_op(&read_op, read_op_dest());
    let make_opaque_op = BoolOp::make_opaque.to_extension_op().unwrap();
    lw.replace_op(&make_opaque_op, make_opaque_op_dest());
    for op in [BoolOp::eq, BoolOp::and, BoolOp::or, BoolOp::xor] {
        lw.replace_op(&op.to_extension_op().unwrap(), binary_logic_op_dest(&op));
    }
    let not_op = BoolOp::not.to_extension_op().unwrap();
    lw.replace_op(&not_op, not_op_dest());

    // Replace measure ops with lazy versions.
    let tket_measure_free = TketOp::MeasureFree.to_extension_op().unwrap();
    let qsystem_measure = QSystemOp::Measure.to_extension_op().unwrap();
    let qsystem_measure_reset = QSystemOp::MeasureReset.to_extension_op().unwrap();
    lw.replace_op(&tket_measure_free, measure_dest());
    lw.replace_op(&qsystem_measure, measure_dest());
    lw.replace_op(&qsystem_measure_reset, measure_reset_dest());

    // Replace array and borrow ops with copyable bounds with DFGs that the linearizer
    // can act on now that the elements are no longer copyable.
    lw.replace_parametrized_op_with(
        array::EXTENSION.get_op(ARRAY_CLONE_OP_ID.as_str()).unwrap(),
        move |args| {
            let [size, elem_ty] = args else {
                unreachable!()
            };
            let size = size.as_nat().unwrap();
            let elem_ty = elem_ty.as_runtime().unwrap();
            (!elem_ty.copyable()).then(|| {
                let array_ty = array_type(size, elem_ty.clone());
                array_clone_dest(array_ty)
            })
        },
        ReplacementOptions::default().with_linearization(true),
    );

    lw.replace_parametrized_op_with(
        borrow_array::EXTENSION
            .get_op(ARRAY_CLONE_OP_ID.as_str())
            .unwrap(),
        move |args| {
            let [size, elem_ty] = args else {
                unreachable!()
            };
            let size = size.as_nat().unwrap();
            let elem_ty = elem_ty.as_runtime().unwrap();
            (!elem_ty.copyable()).then(|| {
                let array_ty = borrow_array_type(size, elem_ty.clone());
                array_clone_dest(array_ty)
            })
        },
        ReplacementOptions::default().with_linearization(true),
    );

    let drop_op_def = GUPPY_EXTENSION.get_op(DROP_OP_NAME.as_str()).unwrap();

    lw.replace_parametrized_op(
        array::EXTENSION
            .get_op(ARRAY_DISCARD_OP_ID.as_str())
            .unwrap(),
        move |args| {
            let [size, elem_ty] = args else {
                unreachable!()
            };
            let size = size.as_nat().unwrap();
            let elem_ty = elem_ty.as_runtime().unwrap();
            if elem_ty.copyable() {
                return None;
            }
            let drop_op = ExtensionOp::new(
                drop_op_def.clone(),
                vec![array_type(size, elem_ty.clone()).into()],
            )
            .unwrap();
            Some(NodeTemplate::SingleOp(drop_op.into()))
        },
    );

    lw.replace_parametrized_op(
        borrow_array::EXTENSION
            .get_op(ARRAY_DISCARD_OP_ID.as_str())
            .unwrap(),
        move |args| {
            let [size, elem_ty] = args else {
                unreachable!()
            };
            let size = size.as_nat().unwrap();
            let elem_ty = elem_ty.as_runtime().unwrap();
            if elem_ty.copyable() {
                return None;
            }
            let drop_op = ExtensionOp::new(
                drop_op_def.clone(),
                vec![borrow_array_type(size, elem_ty.clone()).into()],
            )
            .unwrap();
            Some(NodeTemplate::SingleOp(drop_op.into()))
        },
    );

    lw
}

#[cfg(test)]
mod test {
    use crate::extension::qsystem::{QSystemOp, QSystemOpBuilder};

    use super::*;
    use hugr::ops::OpType;
    use hugr::std_extensions::collections::borrow_array::BArrayOpBuilder;
    use hugr::type_row;
    use hugr::{
        builder::{inout_sig, DFGBuilder, Dataflow, DataflowHugr},
        extension::prelude::qb_t,
        types::TypeRow,
        HugrView,
    };
    use rstest::rstest;
    use tket::{
        extension::bool::{BoolOp, BoolOpBuilder},
        TketOp,
    };

    fn tket_bool_t() -> Type {
        bool_type().clone()
    }

    #[test]
    fn test_consts() {
        let mut dfb = DFGBuilder::new(inout_sig(vec![], vec![tket_bool_t()])).unwrap();
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
        let mut dfb = DFGBuilder::new(inout_sig(vec![tket_bool_t()], vec![bool_t()])).unwrap();
        let [b] = dfb.input_wires_arr();
        let output = dfb.add_bool_read(b).unwrap();
        let mut h = dfb.finish_hugr_with_outputs(output).unwrap();

        assert_eq!(h.num_nodes(), 8);

        let pass = ReplaceBoolPass;
        pass.run(&mut h).unwrap();
        h.validate().unwrap();

        let sig = h.signature(h.entrypoint()).unwrap();
        assert_eq!(sig.input(), &TypeRow::from(vec![bool_dest()]));
        assert_eq!(sig.output(), &TypeRow::from(vec![bool_t()]));

        assert_eq!(h.num_nodes(), 18);
    }

    #[test]
    fn test_make_opaque() {
        let mut dfb = DFGBuilder::new(inout_sig(vec![bool_t()], vec![tket_bool_t()])).unwrap();
        let [b] = dfb.input_wires_arr();
        let output = dfb.add_bool_make_opaque(b).unwrap();
        let mut h = dfb.finish_hugr_with_outputs(output).unwrap();

        assert_eq!(h.num_nodes(), 8);

        let pass = ReplaceBoolPass;
        pass.run(&mut h).unwrap();
        h.validate().unwrap();

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
            vec![tket_bool_t(), tket_bool_t()],
            vec![tket_bool_t()],
        ))
        .unwrap();
        let [b1, b2] = dfb.input_wires_arr();
        let result = dfb.add_dataflow_op(logic_op, [b1, b2]).unwrap();
        let mut h = dfb.finish_hugr_with_outputs(result.outputs()).unwrap();

        let pass = ReplaceBoolPass;
        pass.run(&mut h).unwrap();
        h.validate().unwrap();

        let sig = h.signature(h.entrypoint()).unwrap();
        assert_eq!(sig.input(), &TypeRow::from(vec![bool_dest(), bool_dest()]));
        assert_eq!(sig.output(), &TypeRow::from(vec![bool_dest()]));
    }

    #[test]
    fn test_not() {
        let mut dfb = DFGBuilder::new(inout_sig(vec![tket_bool_t()], vec![tket_bool_t()])).unwrap();
        let [b] = dfb.input_wires_arr();
        let result = dfb.add_dataflow_op(BoolOp::not, [b]).unwrap();
        let mut h = dfb.finish_hugr_with_outputs(result.outputs()).unwrap();

        let pass = ReplaceBoolPass;
        pass.run(&mut h).unwrap();
        h.validate().unwrap();

        let sig = h.signature(h.entrypoint()).unwrap();
        assert_eq!(sig.input(), &TypeRow::from(vec![bool_dest()]));
        assert_eq!(sig.output(), &TypeRow::from(vec![bool_dest()]));
    }

    #[rstest]
    #[case(TketOp::MeasureFree)]
    #[case(QSystemOp::Measure)]
    fn test_measure<T: Into<OpType>>(#[case] measure_op: T) {
        let mut dfb = DFGBuilder::new(inout_sig(vec![qb_t()], vec![bool_type()])).unwrap();
        let [q] = dfb.input_wires_arr();
        let output = dfb.add_dataflow_op(measure_op, [q]).unwrap();
        let mut h = dfb.finish_hugr_with_outputs(output.outputs()).unwrap();

        let pass = ReplaceBoolPass;
        pass.run(&mut h).unwrap();
        h.validate().unwrap();

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
        h.validate().unwrap();

        let sig = h.signature(h.entrypoint()).unwrap();
        assert_eq!(sig.output(), &TypeRow::from(vec![qb_t(), bool_dest()]));
    }

    #[test]
    fn test_array_clone_bool() {
        let elem_ty = bool_type();
        let size = 4;
        let arr_ty = borrow_array_type(size, elem_ty.clone());
        let mut dfb = DFGBuilder::new(inout_sig(
            vec![arr_ty.clone()],
            vec![arr_ty.clone(), arr_ty.clone()],
        ))
        .unwrap();
        let [arr_in] = dfb.input_wires_arr();
        let (arr1, arr2) = dfb.add_borrow_array_clone(elem_ty, size, arr_in).unwrap();
        let mut h = dfb.finish_hugr_with_outputs([arr1, arr2]).unwrap();

        h.validate().unwrap();
        let pass = ReplaceBoolPass;
        pass.run(&mut h).unwrap();
        h.validate().unwrap();

        let sig = h.signature(h.entrypoint()).unwrap();
        let bool_dest_ty = bool_dest();
        let arr_dest_ty = borrow_array_type(size, bool_dest_ty);
        assert_eq!(sig.input(), &TypeRow::from(vec![arr_dest_ty.clone()]));
        assert_eq!(
            sig.output(),
            &TypeRow::from(vec![arr_dest_ty.clone(), arr_dest_ty])
        );
    }

    #[test]
    fn test_array_discard_bool() {
        let elem_ty = bool_type();
        let size = 4;
        let arr_ty = borrow_array_type(size, elem_ty.clone());
        let mut dfb = DFGBuilder::new(inout_sig(vec![arr_ty.clone()], type_row![])).unwrap();
        let [arr_in] = dfb.input_wires_arr();
        dfb.add_borrow_array_discard(elem_ty, size, arr_in).unwrap();
        let mut h = dfb.finish_hugr_with_outputs([]).unwrap();

        h.validate().unwrap();
        let pass = ReplaceBoolPass;
        pass.run(&mut h).unwrap();
        h.validate().unwrap();
    }
}
