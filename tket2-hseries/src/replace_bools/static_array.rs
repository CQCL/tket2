use hugr::{
    algorithms::{
        replace_types::{NodeTemplate, ReplaceTypesError},
        ComposablePass, ReplaceTypes,
    }, builder::{
        inout_sig, BuildError, DFGBuilder, Dataflow, DataflowHugr as _, DataflowSubContainer as _,
        SubContainer as _,
    }, extension::{
        prelude::{bool_t, option_type, usize_t},
        simple_op::{HasConcrete as _, MakeOpDef},
    }, hugr::hugrmut::HugrMut, ops::{OpTrait as _, Tag, Value}, std_extensions::collections::{
        array::ArrayValue,
        static_array::{
            self, static_array_type, StaticArrayOpBuilder as _, StaticArrayOpDef, StaticArrayValue,
            STATIC_ARRAY_TYPENAME,
        },
    }, types::{Transformable as _, Type, TypeEnum, TypeRow}, HugrView as _, Node, Wire
};
use itertools::Itertools as _;
use tket2::extension::bool::{self, bool_type, BoolOpBuilder as _, ConstBool, BOOL_TYPE_NAME};

#[non_exhaustive]
#[derive(Debug, derive_more::Error, derive_more::Display, derive_more::From)]
pub enum LowerStaticArrayBoolPassError {
    ReplaceTypesError(ReplaceTypesError),
    BuildError(BuildError),
}

type Result<T> = std::result::Result<T, LowerStaticArrayBoolPassError>;

pub struct LowerStaticArrayBoolPass (ReplaceTypes);

impl<H: HugrMut<Node=Node>> ComposablePass<H> for LowerStaticArrayBoolPass {
    type Error = LowerStaticArrayBoolPassError;
    type Result = bool;

    fn run(&self, hugr: &mut H) -> Result<bool> {
        Ok(self.0.run(hugr)?)
    }
}

impl LowerStaticArrayBoolPass {
    pub fn new() -> Self {
        let inner = {
            let mut inner = ReplaceTypes::default();
            register_const_static_array(&mut inner, None);
            inner.replace_type(bool_type().as_extension().unwrap().clone(), bool_t());
            inner.replace_consts(
                bool_type().as_extension().unwrap().clone(),
                |const_bool, _| {
                    let cb: &ConstBool = const_bool.value().downcast_ref::<ConstBool>().unwrap();
                    Ok(Value::from_bool(cb.value()))
                },
            );
            inner
        };

        let outer = {
            let mut outer = ReplaceTypes::default();
            register_const_static_array(&mut outer, Some(inner.clone()));

            let static_array_typedef = static_array::EXTENSION
                .get_type(&STATIC_ARRAY_TYPENAME)
                .unwrap();
            outer.replace_parametrized_type(static_array_typedef, {
                let inner = inner.clone();
                move |args| {
                let mut element_ty = {
                    let [element_ty_arg] = args else {
                        unreachable!()
                    };
                    element_ty_arg.as_type().unwrap()
                };
                let changed = element_ty.transform(&inner).unwrap();
                changed.then_some(static_array_type(element_ty))
            }});
            outer.replace_parametrized_op(
                static_array::EXTENSION
                    .get_op(&StaticArrayOpDef::get.opdef_id())
                    .unwrap(), {
                        let inner = inner.clone();
                    move |args| {
                        let [element_ty] = args else { unreachable!() };
                        get_op_dest(&inner, element_ty.as_type().unwrap())
                    }
                    });
            outer.replace_parametrized_op(
                static_array::EXTENSION
                    .get_op(&StaticArrayOpDef::len.opdef_id())
                    .unwrap(),
                move |args| {
                    let [element_ty] = args else { unreachable!() };
                    len_op_dest(&inner, element_ty.as_type().unwrap())
                },
            );
            outer
        };
        Self(outer)
    }
}

fn register_const_static_array(rt: &mut ReplaceTypes, inner_rt: Option<ReplaceTypes>) {
    let static_array_typedef = static_array::EXTENSION
        .get_type(&STATIC_ARRAY_TYPENAME)
        .unwrap();
    rt.replace_consts_parametrized(static_array_typedef, move |opaque_val, rt| {
        let rt = inner_rt.as_ref().unwrap_or(rt);
        let Some(mut sav): Option<StaticArrayValue> = opaque_val
            .value()
            .downcast_ref::<StaticArrayValue>()
            .cloned() else {
            return Ok(None)
        };
        let mut any_changed = false;
        let values = sav.value.get_contents().iter().cloned().map(|mut v| {
            any_changed |= rt.change_value(&mut v).unwrap();
            v
        }).collect_vec();
        if !any_changed {
            return Ok(None);
        }
        let element_ty = {
            let mut t = sav.get_element_type().clone();
            t.transform(rt).unwrap();
            t
        };
        sav.value = ArrayValue::new(element_ty, values);
        Ok(Some(sav.into()))
    });
}

fn len_op_dest(rt: &ReplaceTypes, mut elem_ty: Type) -> Option<NodeTemplate> {
    let changed = elem_ty.transform(rt).unwrap();
    changed.then_some(NodeTemplate::SingleOp(
        StaticArrayOpDef::len
            .instantiate(&[elem_ty.into()])
            .unwrap()
            .into(),
    ))
}

fn build_new_to_old(rt: &ReplaceTypes, builder: &mut impl Dataflow, val: Wire, old_ty: Type) -> Result<Wire> {
    match old_ty.as_type_enum() {
        TypeEnum::Extension(custom_ty) => {
            if (custom_ty.extension(), custom_ty.name())
                == (&static_array::EXTENSION_ID, &STATIC_ARRAY_TYPENAME)
            {
                Ok(val)
            } else if (custom_ty.extension(), custom_ty.name())
                == (&bool::BOOL_EXTENSION_ID, &BOOL_TYPE_NAME)
            {
                let [val] = builder.add_bool_make_opaque(val)?;
                Ok(val)
            } else {
                Ok(val)
            }
        }
        TypeEnum::Sum(sum_ty) => {
            let mut new_sum_ty = sum_ty.clone();

            if !new_sum_ty.transform(rt)? {
                Ok(val)
            } else {
                let new_variants: Vec<TypeRow> = new_sum_ty
                    .variants()
                    .cloned()
                    .map(|x| x.try_into().unwrap())
                    .collect_vec();
                let old_variants: Vec<TypeRow> = sum_ty
                    .variants()
                    .cloned()
                    .map(|x| x.try_into().unwrap())
                    .collect_vec();
                let mut cond = builder
                    .conditional_builder(
                        (new_variants.iter().cloned(), val),
                        [],
                        vec![old_ty].into(),
                    )
                    .unwrap();
                for (i, v) in old_variants.iter().enumerate() {
                    let mut case = cond.case_builder(i).unwrap();
                    let ws: Vec<_> = v
                        .iter()
                        .zip(case.input_wires())
                        .map(|(t, w)| build_new_to_old(rt, &mut case, w, t.clone()))
                        .try_collect()?;
                    let [r] = case
                        .add_dataflow_op(Tag::new(i, old_variants.clone()), ws)?
                        .outputs_arr();
                    case.finish_with_outputs([r])?;
                }
                let [r] = cond.finish_sub_container()?.outputs_arr();
                Ok(r)
            }
        }
        _ => panic!("dougrulz"),
    }
}

fn get_op_dest(rt: &ReplaceTypes, old_elem_ty: Type) -> Option<NodeTemplate> {
    let hugr = {
        let mut hugr1 = {
            let mut dfb = DFGBuilder::new(inout_sig(
                vec![static_array_type(old_elem_ty.clone()), usize_t()],
                Type::from(option_type(old_elem_ty.clone())),
            ))
            .unwrap();
            let [arr, index] = dfb.input_wires_arr();
            let r = dfb
                .add_static_array_get(old_elem_ty.clone(), arr, index)
                .unwrap();
            dfb.finish_hugr_with_outputs([r]).unwrap()
        };
        if !rt.run(&mut hugr1).unwrap() {
            None?
        }
        let new_arr_ty = hugr1
            .entrypoint_optype()
            .dataflow_signature()
            .unwrap()
            .input()[0]
            .clone();

        let res_ty = Type::from(option_type(old_elem_ty.clone()));
        let mut dfb =
            DFGBuilder::new(inout_sig(vec![new_arr_ty, usize_t()], res_ty.clone())).unwrap();
        let [arr, index] = dfb.input_wires_arr();
        let [val] = dfb
            .add_hugr_with_wires(hugr1, [arr, index])
            .unwrap()
            .outputs_arr();

        let old = build_new_to_old(rt, &mut dfb, val, res_ty.clone()).unwrap();
        dfb.finish_hugr_with_outputs([old]).unwrap()
        // dfb.finish_hugr_with_outputs([val]).unwrap()
    };
    Some(NodeTemplate::CompoundOp(Box::new(hugr)))
}


#[cfg(test)]
mod test {
    use hugr::types::SumType;
    use hugr::{algorithms::ComposablePass as _, extension::prelude::option_type, HugrView as _};
    use hugr::{
        builder::DataflowHugr as _, extension::prelude::ConstUsize,
        std_extensions::collections::static_array::StaticArrayOpBuilder as _,
    };
    use hugr::{
        builder::{DFGBuilder, Dataflow},
        extension::prelude::usize_t,
        std_extensions::collections::static_array::StaticArrayValue,
        type_row,
        types::Signature,
    };
    use rstest::rstest;

    use super::*;

    fn static_array_tket2_bool() -> StaticArrayValue {
        StaticArrayValue::try_new("arr", bool_type(), [ConstBool::new(true).into()]).unwrap()
    }

    fn static_array_static_array_tket2_bool() -> StaticArrayValue {
        StaticArrayValue::try_new(
            "arr",
            static_array_type(bool_type()),
            [static_array_tket2_bool().into()],
        )
        .unwrap()
    }

    fn static_array_sum() -> StaticArrayValue {
        let sum_ty = SumType::new(vec![vec![], vec![bool_type()], vec![bool_t()]]);
        StaticArrayValue::try_new(
            "arr",
            sum_ty.clone().into(),
            [Value::sum(2, [Value::false_val()], sum_ty).unwrap()],
        )
        .unwrap()
    }

    #[rstest]
    #[case(static_array_tket2_bool())]
    #[case(static_array_static_array_tket2_bool())]
    #[case(static_array_sum())]
    fn test_static_array_bool_lowerer(#[case] x: StaticArrayValue) {
        let mut hugr = {
            let element_ty = x.get_element_type().clone();
            let mut builder = DFGBuilder::new(Signature::new(
                type_row![],
                vec![option_type(element_ty.clone()).into(), usize_t()],
            ))
            .unwrap();

            let arr = builder.add_load_value(x);

            let zero = builder.add_load_value(ConstUsize::new(0));
            let elem = builder
                .add_static_array_get(element_ty.clone(), arr, zero)
                .unwrap();
            let len = builder.add_static_array_len(element_ty, arr).unwrap();
            builder.finish_hugr_with_outputs([elem, len]).unwrap()
        };

        LowerStaticArrayBoolPass::new().run(&mut hugr).unwrap();
        hugr.validate().unwrap_or_else(|e| panic!("{e}"));
    }
}
