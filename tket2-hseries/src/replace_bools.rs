use derive_more::{Display, Error, From};
use hugr::{
    algorithms::{
        ensure_no_nonlocal_edges,
        non_local::NonLocalEdgesError,
        replace_types::{NodeTemplate, ReplaceTypesError},
        validation::{ValidatePassError, ValidationLevel},
        ReplaceTypes,
    },
    builder::{
        inout_sig, BuildHandle, DFGBuilder, Dataflow, DataflowHugr,
        DataflowSubContainer, SubContainer,
    },
    extension::prelude::bool_t,
    hugr::hugrmut::HugrMut,
    ops::{
        handle::ConditionalID,
        ExtensionOp, Tag,
    },
    std_extensions::logic::LogicOp,
    types::{SumType, Type},
    Hugr, Node, Wire,
};
use tket2::extension::bool::{bool_type, BOOL_EXTENSION};

use crate::extension::futures::{future_type, FutureOpBuilder};

#[derive(Error, Debug, Display, From)]
#[non_exhaustive]
/// An error reported from [ReplaceBoolPass].
pub enum ReplaceBoolPassError<N> {
    /// The HUGR was invalid either before or after a pass ran.
    ValidationError(ValidatePassError),
    /// The HUGR was found to contain non-local edges.
    NonLocalEdgesError(NonLocalEdgesError<N>),
    /// There was an error while replacing the type.
    ReplacementError(ReplaceTypesError),
}

/// A HUGR -> HUGR pass that replaces the tket2.bool type with a sum type of bool_t and
/// future (bool_t) in order to enable lazy measurements.
#[derive(Default)]
pub struct ReplaceBoolPass(ValidationLevel);

impl ReplaceBoolPass {
    /// Run `ReplaceBoolPass` on the given [HugrMut].
    pub fn run(&self, hugr: &mut impl HugrMut) -> Result<(), ReplaceBoolPassError<Node>> {
        self.0.run_validated_pass(hugr, |hugr, level| {
            if *level != ValidationLevel::None {
                ensure_no_nonlocal_edges(hugr)?;
            }
            lowerer().run(hugr)?;
            Ok(())
        })
    }

    /// Returns a new `ReplaceBoolPass` with the given [ValidationLevel].
    pub fn with_validation_level(mut self, level: ValidationLevel) -> Self {
        self.0 = level;
        self
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

/// The configuration used for replacing tket2.bool extension types and ops.
fn lowerer() -> ReplaceTypes {
    let mut lw = ReplaceTypes::default();

    // Replace tket2.bool type.
    lw.replace_type(bool_type().as_extension().unwrap().clone(), bool_dest());

    // Replace al tket2.bool ops.
    let read_op = ExtensionOp::new(
        BOOL_EXTENSION.get_op("read").unwrap().clone(),
        vec![],
    )
    .unwrap();
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

    // TODO: Replace measure ops with lazy versions.

    lw
}

#[cfg(test)]
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
    fn test_read() {
        let mut dfb = DFGBuilder::new(inout_sig(vec![tket2_bool_t()], vec![bool_t()])).unwrap();
        let [b] = dfb.input_wires_arr();
        let output = dfb.add_bool_read(b).unwrap();
        let mut h = dfb.finish_hugr_with_outputs(output).unwrap();

        let pass = ReplaceBoolPass::default();
        pass.run(&mut h).unwrap();

        let sig = h.signature(h.root()).unwrap();
        assert_eq!(sig.input(), &TypeRow::from(vec![bool_dest()]));
        assert_eq!(sig.output(), &TypeRow::from(vec![bool_t()]));
    }

    #[test]
    fn test_make_opaque() {
        let mut dfb = DFGBuilder::new(inout_sig(vec![bool_t()], vec![tket2_bool_t()])).unwrap();
        let [b] = dfb.input_wires_arr();
        let output = dfb.add_make_bool_opaque(b).unwrap();
        let mut h = dfb.finish_hugr_with_outputs(output).unwrap();

        let pass = ReplaceBoolPass::default();
        pass.run(&mut h).unwrap();

        let sig = h.signature(h.root()).unwrap();
        assert_eq!(sig.input(), &TypeRow::from(vec![bool_t()]));
        assert_eq!(sig.output(), &TypeRow::from(vec![bool_dest()]));
    }

    #[test]
    fn test_logic() {
        let mut dfb = DFGBuilder::new(inout_sig(
            vec![tket2_bool_t(), tket2_bool_t()],
            vec![tket2_bool_t()],
        ))
        .unwrap();
        let [b1, b2] = dfb.input_wires_arr();
        let result = dfb.add_dataflow_op(BoolOp::xor, [b1, b2]).unwrap();
        let mut h = dfb.finish_hugr_with_outputs(result.outputs()).unwrap();

        let pass = ReplaceBoolPass::default();
        pass.run(&mut h).unwrap();

        let sig = h.signature(h.root()).unwrap();
        assert_eq!(sig.input(), &TypeRow::from(vec![bool_dest(), bool_dest()]));
        assert_eq!(sig.output(), &TypeRow::from(vec![bool_dest()]));
    }
}
