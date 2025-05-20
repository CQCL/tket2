use hugr::{
    algorithms::{
        replace_types::{NodeTemplate, ReplaceTypesError},
        ComposablePass as _, ReplaceTypes,
    },
    builder::{
        inout_sig, BuildError, DFGBuilder, Dataflow, DataflowHugr as _, DataflowSubContainer as _,
        SubContainer as _,
    },
    extension::{
        prelude::{bool_t, option_type, usize_t},
        simple_op::{HasConcrete as _, MakeOpDef},
    },
    ops::{OpTrait as _, Tag, Value},
    std_extensions::collections::{
        array::ArrayValue,
        static_array::{
            self, static_array_type, StaticArrayOpBuilder as _, StaticArrayOpDef, StaticArrayValue,
            STATIC_ARRAY_TYPENAME,
        },
    },
    types::{SumType, Transformable as _, Type, TypeEnum, TypeRow},
    HugrView as _, Wire,
};
use itertools::Itertools as _;
use tket2::extension::bool::{self, bool_type, BoolOpBuilder as _, ConstBool, BOOL_TYPE_NAME};

#[non_exhaustive]
#[derive(Debug, derive_more::Error, derive_more::Display, derive_more::From)]
pub enum ReplaceBoolStaticArrayError {
    ReplaceTypesError(ReplaceTypesError),
    BuildError(BuildError),
}

type Result<T> = std::result::Result<T, ReplaceBoolStaticArrayError>;

fn map_sum_ty(st: &mut SumType) -> Result<bool> {
    st.transform(&bool_lowerer()).map_err(Into::into)
}

fn map_ty(ty: &mut Type) -> Result<bool> {
    ty.transform(&bool_lowerer()).map_err(Into::into)
}

fn bool_lowerer() -> ReplaceTypes {
    let mut lowerer = ReplaceTypes::default();
    lowerer.replace_type(bool_type().as_extension().unwrap().clone(), bool_t());
    lowerer.replace_consts(
        bool_type().as_extension().unwrap().clone(),
        |const_bool, _| {
            let cb: &ConstBool = const_bool.value().downcast_ref::<ConstBool>().unwrap();
            Ok(Value::from_bool(cb.value()))
        },
    );
    lowerer
}

fn len_op_dest(mut elem_ty: Type) -> Option<NodeTemplate> {
    let changed = map_ty(&mut elem_ty).unwrap();
    changed.then_some(NodeTemplate::SingleOp(
        StaticArrayOpDef::len
            .instantiate(&[elem_ty.into()])
            .unwrap()
            .into(),
    ))
}

fn build_new_to_old(builder: &mut impl Dataflow, val: Wire, old_ty: Type) -> Result<Wire> {
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
            if !map_sum_ty(&mut new_sum_ty)? {
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
                        .map(|(t, w)| build_new_to_old(&mut case, w, t.clone()))
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

fn get_op_dest(old_elem_ty: Type) -> Option<NodeTemplate> {
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
        if !bool_lowerer().run(&mut hugr1).unwrap() {
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

        let old = build_new_to_old(&mut dfb, val, res_ty.clone()).unwrap();
        dfb.finish_hugr_with_outputs([old]).unwrap()
        // dfb.finish_hugr_with_outputs([val]).unwrap()
    };
    Some(NodeTemplate::CompoundOp(Box::new(hugr)))
}

pub fn static_array_bool_lowerer() -> ReplaceTypes {
    let mut lowerer = ReplaceTypes::default();
    let static_array_typedef = static_array::EXTENSION
        .get_type(&STATIC_ARRAY_TYPENAME)
        .unwrap();
    lowerer.replace_parametrized_type(static_array_typedef, |args| {
        let mut element_ty = {
            let [element_ty_arg] = args else {
                unreachable!()
            };
            element_ty_arg.as_type().unwrap()
        };
        let changed = map_ty(&mut element_ty).unwrap();
        changed.then_some(static_array_type(element_ty))
    });

    {
        let inner_lowerer = bool_lowerer();
        lowerer.replace_consts_parametrized(static_array_typedef, move |opaque_val, _| {
            let mut sav: StaticArrayValue = opaque_val
                .value()
                .downcast_ref::<StaticArrayValue>()
                .unwrap()
                .clone();
            let mut element_ty = sav.get_element_type().clone();

            if !map_ty(&mut element_ty).unwrap() {
                return Ok(None);
            }
            sav.value = ArrayValue::new(
                element_ty,
                sav.get_contents().iter().cloned().map(|mut v| {
                    let _ = inner_lowerer.change_value(&mut v).unwrap();
                    v
                }),
            );
            Ok(Some(sav.into()))
        });
    }
    lowerer.replace_parametrized_op(
        static_array::EXTENSION
            .get_op(&StaticArrayOpDef::get.opdef_id())
            .unwrap(),
        move |args| {
            let [element_ty] = args else { unreachable!() };
            get_op_dest(element_ty.as_type().unwrap())
        },
    );
    lowerer.replace_parametrized_op(
        static_array::EXTENSION
            .get_op(&StaticArrayOpDef::len.opdef_id())
            .unwrap(),
        move |args| {
            let [element_ty] = args else { unreachable!() };
            len_op_dest(element_ty.as_type().unwrap())
        },
    );
    lowerer
}

#[cfg(test)]
mod test {
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

        static_array_bool_lowerer().run(&mut hugr).unwrap();
        hugr.validate().unwrap_or_else(|e| panic!("{e}"));
    }
}
