use hugr::{
    algorithms::{replace_types::NodeTemplate, validation::ValidationLevel, ReplaceTypes}, builder::{inout_sig, BuildHandle, ConditionalBuilder, DFGBuilder, Dataflow, DataflowHugr, DataflowSubContainer, SubContainer}, extension::prelude::bool_t, hugr::hugrmut::HugrMut, ops::{handle::{ConditionalID, DataflowOpID}, ExtensionOp, Tag}, std_extensions::logic::LogicOp, types::{CustomType, SumType, Type}, Hugr, Wire
};
use tket2::extension::bool::{bool_type, BOOL_EXTENSION};

use crate::extension::futures::{future_type, FutureOpBuilder};

/// A HUGR -> HUGR pass that replaces the tket2.bool type and ops in as well as
/// standard measurement ops with lazy `tket2.qsystem` measurement ops.
#[derive(Default)]
pub struct ReplaceBoolPass(ValidationLevel);

impl ReplaceBoolPass {
    pub fn run(&self, hugr: &mut impl HugrMut) -> () {
        lowerer().run(hugr);
    }
}

fn bool_dest() -> Type {
    SumType::new([bool_t(), future_type(bool_t())]).into()
}

fn cond_builder(dfb: &mut DFGBuilder<Hugr>, sum_wire: Wire) -> BuildHandle<ConditionalID> {
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

    let cond = cb.finish_sub_container().unwrap();
    cond
}

fn bool_to_sum_dest() -> NodeTemplate {
    let mut dfb = DFGBuilder::new(inout_sig(vec![bool_dest()], vec![bool_t()])).unwrap();
    let [sum_wire] = dfb.input_wires_arr();
    let cond = cond_builder(&mut dfb, sum_wire);
    let h = dfb.finish_hugr_with_outputs(cond.outputs()).unwrap();
    NodeTemplate::CompoundOp(Box::new(h))
}

fn sum_to_bool_dest() -> NodeTemplate {
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
    let mut dfb = DFGBuilder::new(inout_sig(vec![bool_dest(), bool_dest()], vec![bool_dest()])).unwrap();
    let [sum_wire1, sum_wire2] = dfb.input_wires_arr();
    let cond1 = cond_builder(&mut dfb, sum_wire1);
    let cond2 = cond_builder(&mut dfb, sum_wire2);
    let result = match op_name {
        "eq" => dfb.add_dataflow_op(LogicOp::Eq, [cond1.out_wire(0), cond2.out_wire(0)]).unwrap(),
        "and" => dfb.add_dataflow_op(LogicOp::And, [cond1.out_wire(0), cond2.out_wire(0)]).unwrap(),
        "or" => dfb.add_dataflow_op(LogicOp::Or, [cond1.out_wire(0), cond2.out_wire(0)]).unwrap(),
        "xor" => dfb.add_dataflow_op(LogicOp::Xor, [cond1.out_wire(0), cond2.out_wire(0)]).unwrap(),
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
    let cond = cond_builder(&mut dfb, sum_wire);
    let result = dfb.add_dataflow_op(LogicOp::Not, [cond.out_wire(0)]).unwrap();
    let out = dfb
    .add_dataflow_op(
        Tag::new(0, vec![bool_t().into(), future_type(bool_t()).into()]),
        vec![result.out_wire(0)],
    )
    .unwrap();
    let h = dfb.finish_hugr_with_outputs(out.outputs()).unwrap();
    NodeTemplate::CompoundOp(Box::new(h))
}

// The configuration used for replacing tket2.bool extension types and ops.
fn lowerer() -> ReplaceTypes {
    let mut lw = ReplaceTypes::default();

    // Replace type.
    lw.replace_type(bool_type().as_extension().unwrap().clone(), bool_dest());

    // Replace reading and tagging ops.
    let bool_to_sum_op = ExtensionOp::new(
        BOOL_EXTENSION.get_op("bool_to_sum").unwrap().clone(),
        vec![],
    )
    .unwrap();
    lw.replace_op(&bool_to_sum_op, bool_to_sum_dest());
    let sum_to_bool_op = ExtensionOp::new(
        BOOL_EXTENSION.get_op("sum_to_bool").unwrap().clone(),
        vec![],
    )
    .unwrap();
    lw.replace_op(&sum_to_bool_op, sum_to_bool_dest());

    // Replace logic ops.
    for op_name in ["eq", "and", "or", "xor"] {
        let op = ExtensionOp::new(
            BOOL_EXTENSION.get_op(op_name).unwrap().clone(),
            vec![],
        )
        .unwrap();
        lw.replace_op(&op, binary_logic_op_dest(op_name));
    }
    let not_op = ExtensionOp::new(
        BOOL_EXTENSION.get_op("not").unwrap().clone(),
        vec![],
    )
    .unwrap();
    lw.replace_op(&not_op, not_op_dest());

    // TODO: Replace measure ops with lazy versions.

    lw
}

mod test {
    use super::*;
    use hugr::{
        builder::{inout_sig, DFGBuilder, Dataflow, DataflowHugr},
        types::TypeRow,
        HugrView,
    };
    use tket2::extension::bool::{BoolOp, BoolOpBuilder};

    fn tket2_bool_t() -> Type {
        bool_type().clone()
    }

    #[test]
    fn test_bool_to_sum() {
        let mut dfb = DFGBuilder::new(inout_sig(vec![tket2_bool_t()], vec![bool_t()])).unwrap();
        let [b] = dfb.input_wires_arr();
        let output = dfb.add_bool_to_sum(b).unwrap();
        let mut h = dfb.finish_hugr_with_outputs(output).unwrap();

        let pass = ReplaceBoolPass::default();
        pass.run(&mut h);

        let sig = h.signature(h.root()).unwrap();
        assert_eq!(sig.input(), &TypeRow::from(vec![bool_dest()]));
        assert_eq!(sig.output(), &TypeRow::from(vec![bool_t()]));
    }

    #[test]
    fn test_sum_to_bool() {
        let mut dfb = DFGBuilder::new(inout_sig(vec![bool_t()], vec![tket2_bool_t()])).unwrap();
        let [b] = dfb.input_wires_arr();
        let output = dfb.add_sum_to_bool(b).unwrap();
        let mut h = dfb.finish_hugr_with_outputs(output).unwrap();

        let pass = ReplaceBoolPass::default();
        pass.run(&mut h);

        let sig = h.signature(h.root()).unwrap();
        assert_eq!(sig.input(), &TypeRow::from(vec![bool_t()]));
        assert_eq!(sig.output(), &TypeRow::from(vec![bool_dest()]));
    }

    #[test]
    fn test_logic() {
        let mut dfb = DFGBuilder::new(inout_sig(vec![tket2_bool_t(), tket2_bool_t()], vec![tket2_bool_t()])).unwrap();
        let [b1, b2] = dfb.input_wires_arr();
        let result = dfb.add_dataflow_op(BoolOp::xor, [b1, b2]).unwrap();
        let mut h = dfb.finish_hugr_with_outputs(result.outputs()).unwrap();

        let pass = ReplaceBoolPass::default();
        pass.run(&mut h);

        let sig = h.signature(h.root()).unwrap();
        assert_eq!(sig.input(), &TypeRow::from(vec![bool_dest(), bool_dest()]));
        assert_eq!(sig.output(), &TypeRow::from(vec![bool_dest()]));
    }
}
