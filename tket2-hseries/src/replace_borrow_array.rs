//! Provides a `ReplaceBorrowArrayPass` which replaces the `borrow_array<n, T>` type
//! (and corresponding ops) with `array<n, option<T>>`.
use std::vec;

use derive_more::{Display, Error, From};
use hugr::{
    algorithms::{
        non_local::FindNonLocalEdgesError,
        replace_types::{NodeTemplate, ReplaceTypesError},
        ComposablePass, ReplaceTypes,
    },
    builder::{inout_sig, DFGBuilder, Dataflow, DataflowHugr},
    extension::{
        prelude::{const_none, const_some, either_type, option_type, usize_t, UnwrapBuilder},
        simple_op::MakeOpDef,
    },
    hugr::hugrmut::HugrMut,
    ops::{Tag, Value},
    std_extensions::collections::{
        array::{
            array_type, ArrayClone, ArrayDiscard, ArrayOp, ArrayOpBuilder, ArrayOpDef, ArrayRepeat, ArrayScan, ArrayValue, ARRAY_CLONE_OP_ID, ARRAY_DISCARD_OP_ID, ARRAY_REPEAT_OP_ID, ARRAY_SCAN_OP_ID
        },
        borrow_array::{self, BArrayOpDef, BArrayUnsafeOpDef, BArrayValue, BORROW_ARRAY_TYPENAME},
    },
    types::{type_param::SeqPart, Term, Type, TypeArg, TypeRow},
    Node,
};
use strum::IntoEnumIterator;

#[derive(Error, Debug, Display, From)]
#[non_exhaustive]
/// An error reported from [ReplaceBorrowArrayPass].
pub enum ReplaceBorrowArrayPassError<N> {
    /// The HUGR was found to contain non-local edges.
    NonLocalEdgesError(FindNonLocalEdgesError<N>),
    /// There was an error while replacing the type.
    ReplacementError(ReplaceTypesError),
}

/// A HUGR -> HUGR pass which replaces the `borrow_array<n, T>` type (and corresponding
/// ops) with `array<n, option<T>>`.
#[derive(Default, Debug, Clone)]
pub struct ReplaceBorrowArrayPass;

impl<H: HugrMut<Node = Node>> ComposablePass<H> for ReplaceBorrowArrayPass {
    type Error = ReplaceBorrowArrayPassError<H::Node>;
    type Result = ();

    fn run(&self, hugr: &mut H) -> Result<(), Self::Error> {
        // TODO uncomment once https://github.com/CQCL/hugr/issues/1234 is complete
        // ensure_no_nonlocal_edges(hugr)?;
        let lowerer = lowerer();
        lowerer.run(hugr)?;
        Ok(())
    }
}

fn replace_const_borrow_array(
    bav: BArrayValue,
    _: &ReplaceTypes,
) -> std::result::Result<Option<Value>, ReplaceTypesError> {
    let option_elem_ty = option_type(bav.get_element_type().clone());
    // TODO: For now we assume that each element in the array is a value, as this is a
    // pass meant to be run on Guppy-generated HUGRs, which should not emit borrow array
    // constants that contain empty values. The borrow array extension however does not
    // guarantee this, so we should handle the case where the array contains empty
    // values on we add a more specific borrow array value that supports this.
    let option_values: Vec<_> = bav
        .get_contents()
        .into_iter()
        .map(|v| const_some(v.clone()))
        .collect();
    let array_value = ArrayValue::new(option_elem_ty.into(), option_values);
    Ok(Some(array_value.into()))
}

fn borrow_dest(size: u64, elem_ty: Type) -> NodeTemplate {
    let opt_elem_ty = option_type(elem_ty.clone());
    let arr_type = array_type(size, opt_elem_ty.clone().into());
    let mut dfb = DFGBuilder::new(inout_sig(
        vec![arr_type.clone(), usize_t()],
        vec![elem_ty.clone(), arr_type.clone()],
    ))
    .unwrap();
    let [arr, idx] = dfb.input_wires_arr();
    let nothing = dfb.add_load_value(const_none(elem_ty.clone()));
    // Get element (and modified array) and put nothing in its place in the array.
    let result = dfb
        .add_array_set(opt_elem_ty.clone().into(), size, arr, idx, nothing)
        .unwrap();
    // Check operation was successful.
    let result_ty = vec![opt_elem_ty.clone().into(), arr_type.clone()];
    let [opt_elem, out_arr] = dfb
        .build_unwrap_sum(1, either_type(result_ty.clone(), result_ty), result)
        .unwrap();
    // Will panic if the retrieved element has been borrowed before (so is nothing).
    let [out] = dfb
        .build_unwrap_sum(1, opt_elem_ty.clone(), opt_elem)
        .unwrap();
    let h = dfb.finish_hugr_with_outputs([out, out_arr]).unwrap();
    NodeTemplate::CompoundOp(Box::new(h))
}

fn return_dest(size: u64, elem_ty: Type) -> NodeTemplate {
    let opt_elem_ty = option_type(elem_ty.clone());
    let arr_type = array_type(size, opt_elem_ty.clone().into());
    let mut dfb = DFGBuilder::new(inout_sig(
        vec![arr_type.clone(), usize_t(), elem_ty.clone()],
        vec![arr_type.clone()],
    ))
    .unwrap();
    let [arr, idx, elem] = dfb.input_wires_arr();
    // Wrap element into an option and put into the array.
    let opt_elem = dfb.add_dataflow_op(Tag::new(
        1, vec![TypeRow::new(), elem_ty.clone().into()],
    ), [elem]).unwrap().out_wire(0);
    let result = dfb
        .add_array_set(opt_elem_ty.clone().into(), size, arr, idx, opt_elem)
        .unwrap();
    // Check operation was successful.
    let result_ty = vec![opt_elem_ty.clone().into(), arr_type.clone()];
    let [nothing, out_arr] = dfb
        .build_unwrap_sum(1, either_type(result_ty.clone(), result_ty), result)
        .unwrap();
    // Will panic if the index wasn't previously empty (so retrieved element is not nothing).
    let [] = dfb
        .build_unwrap_sum(0, opt_elem_ty, nothing)
        .unwrap();
    let h = dfb.finish_hugr_with_outputs([out_arr]).unwrap();
    NodeTemplate::CompoundOp(Box::new(h))
}

fn discard_all_borrowed_dest(size: u64, elem_ty: Type) -> NodeTemplate {
    let opt_elem_ty = option_type(elem_ty.clone());
    let arr_type = array_type(size, opt_elem_ty.clone().into());
    let mut dfb = DFGBuilder::new(inout_sig(vec![arr_type.clone()], vec![])).unwrap();
    let [arr] = dfb.input_wires_arr();
    // Pop each element from the array, panicking if it is not nothing.
    let mut current_arr = arr;
    let current_size = size;
    for _ in 0..size {
        let result = dfb
            .add_array_pop_left(opt_elem_ty.clone().into(), current_size, current_arr)
            .unwrap();
        let current_size = current_size - 1;
        let new_arr_type = array_type(current_size, opt_elem_ty.clone().into());
        let result_ty = vec![opt_elem_ty.clone().into(), new_arr_type.clone()];
        let [nothing, out_arr] = dfb
            .build_unwrap_sum(1, option_type(result_ty), result)
            .unwrap();
        // Implicitly drop the nothing.
        let [] = dfb
            .build_unwrap_sum(0, opt_elem_ty.clone().into(), nothing)
            .unwrap();
        current_arr = out_arr;
    }
    // Discard the array that is now of size 0.
    dfb.add_array_discard_empty(opt_elem_ty.into(), current_arr).unwrap();
    let h = dfb.finish_hugr_with_outputs([]).unwrap();
    NodeTemplate::CompoundOp(Box::new(h))
}

fn new_all_borrowed_dest(size: u64, elem_ty: Type) -> NodeTemplate {
    let opt_elem_ty = option_type(elem_ty.clone());
    let arr_type = array_type(size, opt_elem_ty.clone().into());
    let mut dfb = DFGBuilder::new(inout_sig(vec![], vec![arr_type.clone()])).unwrap();
    // Create a new array that contains `none` at each index.
    let mut nothings = vec![];
    for _ in 0..size {
        let nothing = dfb.add_load_value(const_none(elem_ty.clone()));
        nothings.push(nothing);
    }
    let arr = dfb.add_new_array(opt_elem_ty.clone().into(), nothings).unwrap();
    let h = dfb.finish_hugr_with_outputs([arr]).unwrap();
    NodeTemplate::CompoundOp(Box::new(h))
}

fn lowerer() -> ReplaceTypes {
    let mut lw = ReplaceTypes::default();

    let borrow_array_typedef = borrow_array::EXTENSION
        .get_type(&BORROW_ARRAY_TYPENAME)
        .unwrap();

    // Replace type: `borrow_array<size, elem_ty>` -> `array<size, option<elem_ty>>`.
    lw.replace_parametrized_type(
        borrow_array_typedef,
        Box::new(|args: &[TypeArg]| {
            let [TypeArg::BoundedNat(size), TypeArg::Runtime(elem_ty)] = args else {
                unreachable!()
            };
            Some(array_type(*size, option_type(elem_ty.clone()).into()))
        }),
    );

    // Replace constants.
    lw.replace_consts_parametrized(borrow_array_typedef, {
        move |val, rt| {
            let Some(bav): Option<BArrayValue> = val.value().downcast_ref::<BArrayValue>().cloned()
            else {
                return Ok(None);
            };
            replace_const_borrow_array(bav, rt)
        }
    });

    // Replace custom borrow array ops.
    lw.replace_parametrized_op(
        borrow_array::EXTENSION
            .get_op(&BArrayUnsafeOpDef::borrow.opdef_id())
            .unwrap(),
        {
            move |args| {
                let [size, elem_ty] = args else {
                    unreachable!()
                };
                Some(borrow_dest(
                    size.as_nat().unwrap(),
                    elem_ty.as_runtime().unwrap(),
                ))
            }
        },
    );
    lw.replace_parametrized_op(
        borrow_array::EXTENSION
            .get_op(&BArrayUnsafeOpDef::r#return.opdef_id())
            .unwrap(),
        {
            move |args| {
                let [size, elem_ty] = args else {
                    unreachable!()
                };
                Some(return_dest(
                    size.as_nat().unwrap(),
                    elem_ty.as_runtime().unwrap(),
                ))
            }
        },
    );
    lw.replace_parametrized_op(
        borrow_array::EXTENSION
            .get_op(&BArrayUnsafeOpDef::discard_all_borrowed.opdef_id())
            .unwrap(),
        {
            move |args| {
                let [size, elem_ty] = args else {
                    unreachable!()
                };
                Some(discard_all_borrowed_dest(
                    size.as_nat().unwrap(),
                    elem_ty.as_runtime().unwrap(),
                ))
            }
        },
    );
    lw.replace_parametrized_op(
        borrow_array::EXTENSION
            .get_op(&BArrayUnsafeOpDef::new_all_borrowed.opdef_id())
            .unwrap(),
        {
            move |args| {
                let [size, elem_ty] = args else {
                    unreachable!()
                };
                Some(new_all_borrowed_dest(
                    size.as_nat().unwrap(),
                    elem_ty.as_runtime().unwrap(),
                ))
            }
        },
    );

    // Replace generic array ops.
    lw.replace_parametrized_op(
        borrow_array::EXTENSION
            .get_op(ARRAY_CLONE_OP_ID.as_str())
            .unwrap(),
        move |args| {
            let [elem_ty, size] = args else {
                unreachable!()
            };
            Some(NodeTemplate::SingleOp(
                ArrayClone::new(option_type(elem_ty.as_runtime().unwrap()).into(), size.as_nat().unwrap())
                    .unwrap()
                    .into(),
            ))
        },
    );

    lw.replace_parametrized_op(
        borrow_array::EXTENSION
            .get_op(ARRAY_DISCARD_OP_ID.as_str())
            .unwrap(),
        move |args| {
            let [elem_ty, size] = args else {
                unreachable!()
            };
            Some(NodeTemplate::SingleOp(
                ArrayDiscard::new(option_type(elem_ty.as_runtime().unwrap()).into(), size.as_nat().unwrap())
                    .unwrap()
                    .into(),
            ))
        },
    );

    lw.replace_parametrized_op(
        borrow_array::EXTENSION
            .get_op(ARRAY_REPEAT_OP_ID.as_str())
            .unwrap(),
        move |args| {
            let [elem_ty, size] = args else {
                unreachable!()
            };
            Some(NodeTemplate::SingleOp(
                ArrayRepeat::new(option_type(elem_ty.as_runtime().unwrap()).into(), size.as_nat().unwrap()).into(),
            ))
        },
    );

    lw.replace_parametrized_op(
        borrow_array::EXTENSION
            .get_op(ARRAY_SCAN_OP_ID.as_str())
            .unwrap(),
        move |args| {
            let [src_ty, tgt_ty, acc_tys, size] = args else {
                unreachable!()
            };
            Some(NodeTemplate::SingleOp(
                ArrayScan::new(
                    option_type(src_ty.as_runtime().unwrap()).into(),
                    option_type(tgt_ty.as_runtime().unwrap()).into(),
                    acc_tys.clone()
                        .into_list_parts()
                        .map(|sp| match sp {
                            SeqPart::Item(t) => t.as_runtime().unwrap(),
                            _ => unreachable!("Expected SeqPart::Item"),
                        })
                        .collect(),
                    size.as_nat().unwrap()).into(),
            ))
        },
    );
    
    fn map_borrow_array_to_array_op(op: &BArrayOpDef) -> ArrayOpDef {
        match op {
            BArrayOpDef::new_array => ArrayOpDef::new_array,
            BArrayOpDef::get => ArrayOpDef::get,
            BArrayOpDef::set => ArrayOpDef::set,
            BArrayOpDef::swap => ArrayOpDef::swap,
            BArrayOpDef::pop_left => ArrayOpDef::pop_left,
            BArrayOpDef::pop_right => ArrayOpDef::pop_right,
            BArrayOpDef::discard_empty => ArrayOpDef::discard_empty,
            BArrayOpDef::unpack => ArrayOpDef::unpack,
            _ => panic!("Unsupported borrow array op: {:?}", op),
        }
    }
    for op in borrow_array::BArrayOpDef::iter() {
        let array_op = map_borrow_array_to_array_op(&op);
        lw.replace_parametrized_op(
            borrow_array::EXTENSION.get_op(&op.opdef_id()).unwrap(),
            move |args| {
                let [size, elem_ty] = args else {
                    unreachable!()
                };
                Some(NodeTemplate::SingleOp(
                    array_op.to_concrete(
                        option_type(elem_ty.as_runtime().unwrap()).into(),
                        size.as_nat().unwrap(),
                    )
                    .into(),
                ))
            },
        );
    }

    lw
}

#[cfg(test)]
mod tests {
    use hugr::{
        extension::prelude::{qb_t, ConstUsize}, std_extensions::collections::borrow_array::{borrow_array_type, BArrayOpBuilder}, types::Signature, HugrView
    };

    use super::*;

    #[test]
    fn test_array_type() {
        let size = 4;
        let elem_ty = qb_t();
        let ba_ty = borrow_array_type(size, elem_ty.clone());
        let dfb = DFGBuilder::new(inout_sig(vec![ba_ty.clone()], vec![ba_ty.clone()])).unwrap();
        let [ba] = dfb.input_wires_arr();
        let h = dfb.finish_hugr_with_outputs([ba]).unwrap();

        let mut h = h;
        let pass = ReplaceBorrowArrayPass;
        pass.run(&mut h).unwrap();
        h.validate().unwrap();

        let expected_ty = array_type(size, option_type(elem_ty).into());
        let sig = h.signature(h.entrypoint()).unwrap();
        assert_eq!(sig.input(), &TypeRow::from(vec![expected_ty.clone()]));
        assert_eq!(sig.output(), &TypeRow::from(vec![expected_ty]));
    }

    #[test]
    fn test_borrow_and_return() {
        let size = 22;
        let elem_ty = qb_t();
        let ba_ty = borrow_array_type(size, elem_ty.clone());
        let mut dfb = DFGBuilder::new(inout_sig(vec![ba_ty.clone()], vec![ba_ty.clone()])).unwrap();
        let [ba] = dfb.input_wires_arr();
        let idx = dfb.add_load_value(ConstUsize::new(11));
        let (el, arr_with_borrow) = dfb
            .add_borrow_array_borrow(elem_ty.clone(), size, ba, idx)
            .unwrap();
        let arr_with_return = dfb
            .add_borrow_array_return(elem_ty, size, arr_with_borrow, idx, el)
            .unwrap();
        let mut h = dfb.finish_hugr_with_outputs([arr_with_return]).unwrap();

        let pass = ReplaceBorrowArrayPass;
        pass.run(&mut h).unwrap();
        h.validate().unwrap();
    }

    #[test]
    fn test_discard_all_borrowed() {
        let size = 1;
        let elem_ty = qb_t();
        let arr_ty = borrow_array_type(size, elem_ty.clone());
        let mut builder = DFGBuilder::new(Signature::new(vec![arr_ty.clone()], vec![qb_t()])).unwrap();
        let idx = builder.add_load_value(ConstUsize::new(0));
        let [arr] = builder.input_wires_arr();
        let (el, arr_with_borrowed) = builder
            .add_borrow_array_borrow(elem_ty.clone(), size, arr, idx)
            .unwrap();
        builder
            .add_discard_all_borrowed(elem_ty, size, arr_with_borrowed)
            .unwrap();
        let mut h =  builder.finish_hugr_with_outputs([el]).unwrap();

        let pass = ReplaceBorrowArrayPass;
        pass.run(&mut h).unwrap();
        h.validate().unwrap();
    }

    #[test]
    fn test_new_all_borrowed() {
        let size = 5;
        let elem_ty = usize_t();
        let arr_ty = borrow_array_type(size, elem_ty.clone());
        let mut builder = DFGBuilder::new(Signature::new(vec![], vec![arr_ty.clone()])).unwrap();
        let arr = builder.add_new_all_borrowed(elem_ty.clone(), size).unwrap();
        let idx = builder.add_load_value(ConstUsize::new(3));
        let val = builder.add_load_value(ConstUsize::new(202));
        let arr_with_return = builder
            .add_borrow_array_return(elem_ty, size, arr, idx, val)
            .unwrap();
        let mut h = builder.finish_hugr_with_outputs([arr_with_return]).unwrap();

        let pass = ReplaceBorrowArrayPass;
        pass.run(&mut h).unwrap();
        h.validate().unwrap();
    }
}
