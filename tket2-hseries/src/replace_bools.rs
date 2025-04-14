use hugr::{
    algorithms::{replace_types::NodeTemplate, validation::ValidationLevel, ReplaceTypes},
    builder::{inout_sig, DFGBuilder, Dataflow, DataflowHugr, DataflowSubContainer, SubContainer},
    extension::prelude::bool_t,
    hugr::hugrmut::HugrMut,
    ops::ExtensionOp,
    types::{CustomType, SumType, Type},
};
use tket2::extension::bool::{bool_type, BOOL_EXTENSION};

use crate::extension::futures::{future_type, FutureOpBuilder};

#[derive(Default)]
pub struct ReplaceBoolPass(ValidationLevel);

impl ReplaceBoolPass {
    pub fn run(&self, hugr: &mut impl HugrMut) -> () {
        lowerer().run(hugr);
    }
}

fn tket2_bool_t() -> Type {
    bool_type().clone()
}

fn tket2_bool_t_custom() -> CustomType {
    bool_type().as_extension().unwrap().clone()
}

fn bool_replacement() -> Type {
    SumType::new([bool_t(), future_type(bool_t())]).into()
}

fn read_replacement() -> NodeTemplate {
    let mut dfb = DFGBuilder::new(inout_sig(vec![bool_replacement()], vec![bool_t()])).unwrap();
    let [inp] = dfb.input_wires_arr();
    let mut cb = dfb
        .conditional_builder(
            ([bool_t().into(), future_type(bool_t()).into()], inp),
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
    let h = dfb.finish_hugr_with_outputs(cond.outputs()).unwrap();
    NodeTemplate::CompoundOp(Box::new(h))
}

fn lowerer() -> ReplaceTypes {
    let mut lw = ReplaceTypes::default();

    // A tket2.bool can either be a hugr bool or a future of a hugr bool.
    lw.replace_type(tket2_bool_t_custom(), bool_replacement());

    // Replace bool conversion ops.
    let bool_to_sum_op = ExtensionOp::new(
        BOOL_EXTENSION.get_op("bool_to_sum").unwrap().clone(),
        vec![],
    )
    .unwrap();
    lw.replace_op(&bool_to_sum_op, read_replacement());

    lw
}

mod test {
    use super::*;
    use hugr::{
        builder::{inout_sig, DFGBuilder, Dataflow, DataflowHugr},
        types::TypeRow,
        HugrView,
    };
    use tket2::extension::bool::BoolOpBuilder;

    #[test]
    fn test_read_replacement() {
        let mut dfb = DFGBuilder::new(inout_sig(vec![tket2_bool_t()], vec![bool_t()])).unwrap();
        let [b] = dfb.input_wires_arr();
        let output = dfb.add_bool_to_sum(b).unwrap();
        let mut h = dfb.finish_hugr_with_outputs(output).unwrap();

        let pass = ReplaceBoolPass::default();
        pass.run(&mut h);

        let sig = h.signature(h.root()).unwrap();
        assert_eq!(sig.input(), &TypeRow::from(vec![bool_replacement()]));
        assert_eq!(sig.output(), &TypeRow::from(vec![bool_t()]));
    }
}
