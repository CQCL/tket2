//! Provides a `ReplaceBorrowArrayPass` which replaces the `borrow_array<n, T>` type
//! (and corresponding ops) with `array<n, option<T>>`.
use std::vec;

use derive_more::{Display, Error, From};
use hugr::{
    algorithms::{
        ensure_no_nonlocal_edges,
        non_local::FindNonLocalEdgesError,
        replace_types::{handlers::copy_discard_array, NodeTemplate, ReplaceTypesError},
        ComposablePass, ReplaceTypes,
    },
    builder::{inout_sig, DFGBuilder, Dataflow, DataflowHugr, DataflowSubContainer, SubContainer},
    extension::{
        prelude::{const_none, const_some, either_type, option_type, usize_t, UnwrapBuilder},
        simple_op::MakeOpDef,
    },
    hugr::hugrmut::HugrMut,
    ops::{Tag, Value},
    std_extensions::collections::{
        array::{
            array_type, array_type_def, ArrayClone, ArrayDiscard, ArrayOpBuilder, ArrayOpDef,
            ArrayRepeat, ArrayScan, ArrayValue, ARRAY_CLONE_OP_ID, ARRAY_DISCARD_OP_ID,
            ARRAY_REPEAT_OP_ID, ARRAY_SCAN_OP_ID,
        },
        borrow_array::{self, BArrayOpDef, BArrayUnsafeOpDef, BArrayValue, BORROW_ARRAY_TYPENAME},
    },
    type_row,
    types::{type_param::SeqPart, FuncTypeBase, RowVariable, Signature, Type, TypeArg, TypeRow},
    Hugr, Node, Wire,
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
        ensure_no_nonlocal_edges(hugr)?;
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
    // values once we add a more specific borrow array value that supports this.
    // See https://github.com/CQCL/hugr/issues/2437.
    let option_values: Vec<_> = bav
        .get_contents()
        .iter()
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

    // Set the element at idx to none, retrieving the previous value.
    let result = dfb
        .add_array_set(opt_elem_ty.clone().into(), size, arr, idx, nothing)
        .unwrap();

    // Unwrap the result: expect that the index was in bounds.
    let result_ty = vec![opt_elem_ty.clone().into(), arr_type.clone()];
    let [opt_elem, out_arr] = dfb
        .build_expect_sum(1, either_type(result_ty.clone(), result_ty), result, |_| {
            String::from("Index out of bounds")
        })
        .unwrap();

    // Panic if the element was already borrowed (none).
    let [out] = dfb
        .build_expect_sum(1, opt_elem_ty.clone(), opt_elem, |_| {
            String::from("Element already borrowed")
        })
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
    let opt_elem = option_wrap(&mut dfb, elem, elem_ty);

    // Set the element at idx to Some(elem), retrieving the previous value.
    let result = dfb
        .add_array_set(opt_elem_ty.clone().into(), size, arr, idx, opt_elem)
        .unwrap();

    // Unwrap the result: expect that index was in bounds.
    let result_ty = vec![opt_elem_ty.clone().into(), arr_type.clone()];
    let [nothing, out_arr] = dfb
        .build_expect_sum(1, either_type(result_ty.clone(), result_ty), result, |_| {
            String::from("Index out of bounds")
        })
        .unwrap();

    // Panic if the slot was not empty (not none).
    let [] = dfb
        .build_expect_sum(0, opt_elem_ty, nothing, |_| String::from("Index not empty"))
        .unwrap();

    let h = dfb.finish_hugr_with_outputs([out_arr]).unwrap();
    NodeTemplate::CompoundOp(Box::new(h))
}

fn discard_all_borrowed_dest(size: u64, elem_ty: Type) -> NodeTemplate {
    let opt_elem_ty = option_type(elem_ty.clone());
    let mut dfb = DFGBuilder::new(inout_sig(
        vec![array_type(size, opt_elem_ty.clone().into())],
        vec![],
    ))
    .unwrap();

    let [arr] = dfb.input_wires_arr();
    // This implicitly discards the array.
    let result = dfb
        .add_array_unpack(opt_elem_ty.clone().into(), size, arr)
        .unwrap();
    for wire in result {
        // Ensure value is none.
        let [] = dfb
            .build_expect_sum(0, opt_elem_ty.clone(), wire, |_| {
                String::from("Index not empty")
            })
            .unwrap();
    }

    let h = dfb.finish_hugr_with_outputs([]).unwrap();
    NodeTemplate::CompoundOp(Box::new(h))
}

fn new_all_borrowed_dest(size: u64, elem_ty: Type) -> NodeTemplate {
    let opt_elem_ty = option_type(elem_ty.clone());
    let arr_type = array_type(size, opt_elem_ty.clone().into());
    let mut dfb = DFGBuilder::new(inout_sig(vec![], vec![arr_type.clone()])).unwrap();

    // Fill array with nones.
    let nothings: Vec<_> = (0..size)
        .map(|_| dfb.add_load_value(const_none(elem_ty.clone())))
        .collect();

    let arr = dfb
        .add_new_array(opt_elem_ty.clone().into(), nothings)
        .unwrap();

    let h = dfb.finish_hugr_with_outputs([arr]).unwrap();
    NodeTemplate::CompoundOp(Box::new(h))
}

// Helper function to wrap an element in an option type.
fn option_wrap(dfb: &mut DFGBuilder<Hugr>, elem_wire: Wire, elem_ty: Type) -> Wire {
    dfb.add_dataflow_op(
        Tag::new(1, vec![TypeRow::new(), elem_ty.clone().into()]),
        [elem_wire],
    )
    .unwrap()
    .out_wire(0)
}

fn new_array_dest(size: u64, elem_ty: Type) -> NodeTemplate {
    let opt_elem_ty = option_type(elem_ty.clone());
    let arr_type = array_type(size, opt_elem_ty.clone().into());
    let mut dfb = DFGBuilder::new(inout_sig(
        vec![elem_ty.clone(); size as usize],
        vec![arr_type.clone()],
    ))
    .unwrap();

    // Wrap all input elements in options and collect them.
    let elems = dfb
        .input_wires()
        .map(|wire| option_wrap(&mut dfb, wire, elem_ty.clone()))
        .collect::<Vec<_>>();

    let arr = dfb.add_new_array(opt_elem_ty.into(), elems).unwrap();

    let h = dfb.finish_hugr_with_outputs([arr]).unwrap();
    NodeTemplate::CompoundOp(Box::new(h))
}

fn get_dest(size: u64, elem_ty: Type) -> NodeTemplate {
    let opt_elem_ty = option_type(elem_ty.clone());
    let arr_ty = array_type(size, opt_elem_ty.clone().into());
    let mut dfb = DFGBuilder::new(inout_sig(
        vec![arr_ty.clone(), usize_t()],
        vec![opt_elem_ty.clone().into(), arr_ty.clone()],
    ))
    .unwrap();

    let [arr, idx] = dfb.input_wires_arr();
    let (opt_elem_result, arr_out) = dfb
        .add_array_get(opt_elem_ty.clone().into(), size, arr, idx)
        .unwrap();

    let variant_types = vec![type_row![], vec![opt_elem_ty.clone().into()].into()];
    let mut cond = dfb
        .conditional_builder(
            (variant_types, opt_elem_result),
            [],
            vec![opt_elem_ty.clone().into()].into(),
        )
        .unwrap();

    // Case 0: Out of bounds, return none of element type.
    let mut case_out_of_bounds = cond.case_builder(0).unwrap();
    let none_elem = case_out_of_bounds.add_load_value(const_none(elem_ty.clone()));
    case_out_of_bounds.finish_with_outputs([none_elem]).unwrap();

    // Case 1: In bounds, check that element isn't none and then return it wrapped in option.
    let mut case_in_bounds = cond.case_builder(1).unwrap();
    let [opt_elem] = case_in_bounds.input_wires_arr();
    let [elem] = case_in_bounds
        .build_expect_sum(1, opt_elem_ty.clone(), opt_elem, |_| {
            String::from("Element already borrowed")
        })
        .unwrap();
    let opt_checked_elem = case_in_bounds
        .add_dataflow_op(
            Tag::new(1, vec![TypeRow::new(), elem_ty.clone().into()]),
            [elem],
        )
        .unwrap()
        .out_wire(0);
    case_in_bounds
        .finish_with_outputs([opt_checked_elem])
        .unwrap();

    let [elem_out] = cond.finish_sub_container().unwrap().outputs_arr();
    let h = dfb.finish_hugr_with_outputs([elem_out, arr_out]).unwrap();
    NodeTemplate::CompoundOp(Box::new(h))
}

fn set_dest(size: u64, elem_ty: Type) -> NodeTemplate {
    let opt_elem_ty = option_type(elem_ty.clone());
    let arr_type = array_type(size, opt_elem_ty.clone().into());
    let mut dfb = DFGBuilder::new(inout_sig(
        vec![arr_type.clone(), usize_t(), elem_ty.clone()],
        vec![either_type(
            vec![elem_ty.clone(), arr_type.clone()],
            vec![elem_ty.clone(), arr_type.clone()],
        )
        .into()],
    ))
    .unwrap();

    let [arr, idx, elem] = dfb.input_wires_arr();
    let opt_elem = option_wrap(&mut dfb, elem, elem_ty.clone());

    // Set the element at idx to Some(elem), retrieving the previous value.
    let result = dfb
        .add_array_set(opt_elem_ty.clone().into(), size, arr, idx, opt_elem)
        .unwrap();

    // Check whether the operation was successful or not.
    let mut cond = dfb
        .conditional_builder(
            (
                vec![
                    TypeRow::from(vec![opt_elem_ty.clone().into(), arr_type.clone()]),
                    TypeRow::from(vec![opt_elem_ty.clone().into(), arr_type.clone()]),
                ],
                result,
            ),
            [],
            vec![either_type(
                vec![elem_ty.clone(), arr_type.clone()],
                vec![elem_ty.clone(), arr_type.clone()],
            )
            .into()]
            .into(),
        )
        .unwrap();

    // Case 0: Out of bounds, so return initial element and array after unwrapping.
    let mut case0 = cond.case_builder(0).unwrap();
    let [opt_elem, arr] = case0.input_wires_arr();
    let [prev_elem] = case0
        .build_expect_sum(1, opt_elem_ty.clone(), opt_elem, |_| {
            String::from("Element already borrowed (use return)")
        })
        .unwrap();
    let either = case0
        .add_dataflow_op(
            Tag::new(
                0,
                vec![
                    vec![elem_ty.clone(), arr_type.clone()].into(),
                    vec![elem_ty.clone(), arr_type.clone()].into(),
                ],
            ),
            [prev_elem, arr],
        )
        .unwrap()
        .out_wire(0);
    case0.finish_with_outputs([either]).unwrap();

    // Case 1: Success, so return new element and array after unwrapping.
    let mut case1 = cond.case_builder(1).unwrap();
    let [opt_elem, arr] = case1.input_wires_arr();
    let [new_elem] = case1
        .build_expect_sum(1, opt_elem_ty.clone(), opt_elem, |_| {
            String::from("Element already borrowed (use return)")
        })
        .unwrap();
    let either = case1
        .add_dataflow_op(
            Tag::new(
                1,
                vec![
                    vec![elem_ty.clone(), arr_type.clone()].into(),
                    vec![elem_ty.clone(), arr_type.clone()].into(),
                ],
            ),
            [new_elem, arr],
        )
        .unwrap()
        .out_wire(0);
    case1.finish_with_outputs([either]).unwrap();

    let [final_either] = cond.finish_sub_container().unwrap().outputs_arr();
    let h = dfb.finish_hugr_with_outputs([final_either]).unwrap();
    NodeTemplate::CompoundOp(Box::new(h))
}

fn pop_dest(size: u64, elem_ty: Type, left: bool) -> NodeTemplate {
    let opt_elem_ty = option_type(elem_ty.clone());
    let result_arr_ty = array_type(size - 1, opt_elem_ty.clone().into());
    let opt_result_ty = option_type(vec![elem_ty.clone(), result_arr_ty.clone()]);
    let mut dfb = DFGBuilder::new(inout_sig(
        vec![array_type(size, opt_elem_ty.clone().into())],
        vec![opt_result_ty.clone().into()],
    ))
    .unwrap();

    // Add either pop left or right operation based on whether `left` is true.
    let [arr] = dfb.input_wires_arr();
    let opt_result = if left {
        dfb.add_array_pop_left(opt_elem_ty.clone().into(), size, arr)
            .unwrap()
    } else {
        dfb.add_array_pop_right(opt_elem_ty.clone().into(), size, arr)
            .unwrap()
    };

    let variant_types = vec![
        type_row![],
        vec![opt_elem_ty.clone().into(), result_arr_ty.clone()].into(),
    ];
    let mut cond = dfb
        .conditional_builder(
            (variant_types, opt_result),
            [],
            vec![opt_result_ty.clone().into()].into(),
        )
        .unwrap();

    // Case 0: Popping not successful, return none.
    let mut case0 = cond.case_builder(0).unwrap();
    let none_elem = case0.add_load_value(const_none(vec![elem_ty.clone(), result_arr_ty.clone()]));
    case0.finish_with_outputs([none_elem]).unwrap();

    // Case 1: Popping successful, unwrap the inner option (assume it is always some) and wrap the outer option again.
    let mut case1 = cond.case_builder(1).unwrap();
    let [opt_elem, arr] = case1.input_wires_arr();
    let [elem] = case1
        .build_expect_sum(1, opt_elem_ty.clone(), opt_elem, |_| {
            String::from("Element borrowed")
        })
        .unwrap();
    let new_opt_elem = case1
        .add_dataflow_op(
            Tag::new(
                1,
                vec![
                    TypeRow::new(),
                    TypeRow::from(vec![elem_ty.clone(), result_arr_ty.clone()]),
                ],
            ),
            [elem, arr],
        )
        .unwrap()
        .out_wire(0);
    case1.finish_with_outputs([new_opt_elem]).unwrap();

    let [elem_out] = cond.finish_sub_container().unwrap().outputs_arr();
    let h = dfb.finish_hugr_with_outputs([elem_out]).unwrap();
    NodeTemplate::CompoundOp(Box::new(h))
}

fn unpack_dest(size: u64, elem_ty: Type) -> NodeTemplate {
    let opt_elem_ty = option_type(elem_ty.clone());
    let arr_type = array_type(size, opt_elem_ty.clone().into());
    let mut dfb = DFGBuilder::new(inout_sig(
        vec![arr_type.clone()],
        vec![elem_ty.clone(); size as usize],
    ))
    .unwrap();

    let [arr] = dfb.input_wires_arr();
    let elems = dfb
        .add_array_unpack(opt_elem_ty.clone().into(), size, arr)
        .unwrap();

    // Unwrap each option.
    let mut unwrapped_elems = Vec::with_capacity(size as usize);
    for elem in elems {
        let [unwrapped] = dfb
            .build_expect_sum(1, opt_elem_ty.clone(), elem, |_| {
                String::from("Element borrowed")
            })
            .unwrap();
        unwrapped_elems.push(unwrapped);
    }

    let h = dfb.finish_hugr_with_outputs(unwrapped_elems).unwrap();
    NodeTemplate::CompoundOp(Box::new(h))
}

fn repeat_dest(size: u64, elem_ty: Type) -> NodeTemplate {
    let opt_elem_ty = option_type(elem_ty.clone());
    let arr_type = array_type(size, opt_elem_ty.clone().into());
    let mut dfb = DFGBuilder::new(inout_sig(
        vec![Type::new_function(Signature::new(
            vec![],
            vec![elem_ty.clone()],
        ))],
        vec![arr_type.clone()],
    ))
    .unwrap();

    let [f] = dfb.input_wires_arr();

    let repeat_op = ArrayRepeat::new(elem_ty.clone(), size);

    // Call array_repeat to get array of elements without options.
    let arr = dfb.add_dataflow_op(repeat_op, [f]).unwrap().out_wire(0);

    // Unpack the array to get the elements as wires
    let elems = dfb.add_array_unpack(elem_ty.clone(), size, arr).unwrap();

    // Wrap each element in option
    let opt_elems = elems
        .into_iter()
        .map(|wire| option_wrap(&mut dfb, wire, elem_ty.clone()))
        .collect::<Vec<_>>();

    // Create the new array of option<elem_ty>
    let opt_arr = dfb.add_new_array(opt_elem_ty.into(), opt_elems).unwrap();

    let h = dfb.finish_hugr_with_outputs([opt_arr]).unwrap();
    NodeTemplate::CompoundOp(Box::new(h))
}

fn scan_dest(size: u64, src_ty: Type, tgt_ty: Type, acc_tys: Vec<Type>) -> NodeTemplate {
    let opt_src_ty = option_type(src_ty.clone());
    let opt_tgt_ty = option_type(tgt_ty.clone());
    let src_arr_type = array_type(size, opt_src_ty.clone().into());
    let tgt_arr_type = array_type(size, opt_tgt_ty.clone().into());

    let mut dfb = DFGBuilder::new(inout_sig(
        {
            let mut v = vec![
                src_arr_type.clone(),
                Type::new_function(FuncTypeBase::<RowVariable>::new(
                    {
                        let mut args = vec![src_ty.clone()];
                        args.extend(acc_tys.clone());
                        TypeRow::from(args)
                    },
                    {
                        let mut rets = vec![tgt_ty.clone()];
                        rets.extend(acc_tys.clone());
                        TypeRow::from(rets)
                    },
                )),
            ];
            v.extend(acc_tys.clone());
            TypeRow::from(v)
        },
        {
            let mut v = vec![tgt_arr_type.clone()];
            v.extend(acc_tys.clone());
            TypeRow::from(v)
        },
    ))
    .unwrap();

    let mut inputs: Vec<Wire> = dfb.input_wires().collect();

    // Unwrap all the options in the input array.
    let elems = dfb
        .add_array_unpack(opt_src_ty.clone().into(), size, inputs[0])
        .unwrap();
    let mut unwrapped_elems = Vec::with_capacity(size as usize);
    for elem in elems {
        let [unwrapped] = dfb
            .build_expect_sum(1, opt_src_ty.clone(), elem, |_| {
                String::from("Element borrowed")
            })
            .unwrap();
        unwrapped_elems.push(unwrapped);
    }
    let arr_unwrapped = dfb.add_new_array(src_ty.clone(), unwrapped_elems).unwrap();

    // Add the scan operation on the unwrapped array.
    let scan_op = ArrayScan::new(src_ty.clone(), tgt_ty.clone(), acc_tys.clone(), size);
    inputs[0] = arr_unwrapped;
    let mut outputs: Vec<Wire> = dfb
        .add_dataflow_op(scan_op, inputs)
        .unwrap()
        .outputs()
        .collect();

    // Wrap each element in the output array in option again.
    let out_elems = dfb
        .add_array_unpack(tgt_ty.clone(), size, outputs[0])
        .unwrap();
    let opt_elems = out_elems
        .into_iter()
        .map(|wire| option_wrap(&mut dfb, wire, tgt_ty.clone()))
        .collect::<Vec<_>>();
    let opt_arr = dfb.add_new_array(opt_tgt_ty.into(), opt_elems).unwrap();
    outputs[0] = opt_arr;

    let h = dfb.finish_hugr_with_outputs(outputs).unwrap();
    NodeTemplate::CompoundOp(Box::new(h))
}

fn to_array_dest(size: u64, elem_ty: Type) -> NodeTemplate {
    let opt_elem_ty = option_type(elem_ty.clone());
    let opt_arr_type = array_type(size, opt_elem_ty.clone().into());
    let arr_type = array_type(size, elem_ty.clone().into());

    let mut dfb = DFGBuilder::new(inout_sig(
        vec![opt_arr_type.clone()],
        vec![arr_type.clone()],
    ))
    .unwrap();

    let [opt_arr] = dfb.input_wires_arr();
    let opt_elems = dfb
        .add_array_unpack(opt_elem_ty.clone().into(), size, opt_arr)
        .unwrap();

    // Unwrap all elements to get a standard array.
    let elems = opt_elems
        .into_iter()
        .map(|wire| {
            let [unwrapped] = dfb
                .build_expect_sum(1, opt_elem_ty.clone(), wire, |_| {
                    String::from("Element borrowed")
                })
                .unwrap();
            unwrapped
        })
        .collect::<Vec<_>>();

    let arr = dfb.add_new_array(elem_ty.clone(), elems).unwrap();

    let h = dfb.finish_hugr_with_outputs([arr]).unwrap();
    NodeTemplate::CompoundOp(Box::new(h))
}

fn from_array_dest(size: u64, elem_ty: Type) -> NodeTemplate {
    let opt_elem_ty = option_type(elem_ty.clone());
    let opt_arr_type = array_type(size, opt_elem_ty.clone().into());
    let arr_type = array_type(size, elem_ty.clone().into());

    let mut dfb = DFGBuilder::new(inout_sig(
        vec![arr_type.clone()],
        vec![opt_arr_type.clone()],
    ))
    .unwrap();

    let [arr] = dfb.input_wires_arr();
    let elems = dfb.add_array_unpack(elem_ty.clone(), size, arr).unwrap();

    // Wrap all elements to get a borrow array.
    let wrapped_elems = elems
        .into_iter()
        .map(|wire| option_wrap(&mut dfb, wire, elem_ty.clone()))
        .collect::<Vec<_>>();

    let opt_arr = dfb
        .add_new_array(opt_elem_ty.clone().into(), wrapped_elems)
        .unwrap();

    let h = dfb.finish_hugr_with_outputs([opt_arr]).unwrap();
    NodeTemplate::CompoundOp(Box::new(h))
}

// Register all op replacements defined above.
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

    // Register the array copy and discard ops.
    lw.linearizer()
        .register_callback(array_type_def(), copy_discard_array);

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
    for op_def in BArrayUnsafeOpDef::iter() {
        lw.replace_parametrized_op(
            borrow_array::EXTENSION.get_op(&op_def.opdef_id()).unwrap(),
            {
                move |args| {
                    let [size, elem_ty] = args else {
                        unreachable!()
                    };
                    let size = size.as_nat().unwrap();
                    let elem_ty = elem_ty.as_runtime().unwrap();
                    Some(match op_def {
                        BArrayUnsafeOpDef::borrow => borrow_dest(size, elem_ty),
                        BArrayUnsafeOpDef::r#return => return_dest(size, elem_ty),
                        BArrayUnsafeOpDef::discard_all_borrowed => {
                            discard_all_borrowed_dest(size, elem_ty)
                        }
                        BArrayUnsafeOpDef::new_all_borrowed => new_all_borrowed_dest(size, elem_ty),
                        _ => panic!("Unsupported BArrayUnsafeOpDef variant: {:?}", op_def),
                    })
                }
            },
        );
    }

    // Replace generic array ops.
    lw.replace_parametrized_op(
        borrow_array::EXTENSION
            .get_op(ARRAY_CLONE_OP_ID.as_str())
            .unwrap(),
        move |args| {
            let [size, elem_ty] = args else {
                unreachable!()
            };
            Some(NodeTemplate::SingleOp(
                ArrayClone::new(
                    option_type(elem_ty.as_runtime().unwrap()).into(),
                    size.as_nat().unwrap(),
                )
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
            let [size, elem_ty] = args else {
                unreachable!()
            };
            Some(NodeTemplate::SingleOp(
                ArrayDiscard::new(
                    option_type(elem_ty.as_runtime().unwrap()).into(),
                    size.as_nat().unwrap(),
                )
                .unwrap()
                .into(),
            ))
        },
    );

    lw.replace_parametrized_op(
        borrow_array::EXTENSION
            .get_op(ARRAY_REPEAT_OP_ID.as_str())
            .unwrap(),
        {
            move |args| {
                let [size, elem_ty] = args else {
                    unreachable!()
                };
                Some(repeat_dest(
                    size.as_nat().unwrap(),
                    elem_ty.as_runtime().unwrap(),
                ))
            }
        },
    );

    lw.replace_parametrized_op(
        borrow_array::EXTENSION
            .get_op(ARRAY_SCAN_OP_ID.as_str())
            .unwrap(),
        move |args| {
            let [size, src_ty, tgt_ty, acc_tys] = args else {
                unreachable!()
            };
            Some(scan_dest(
                size.as_nat().unwrap(),
                src_ty.as_runtime().unwrap(),
                tgt_ty.as_runtime().unwrap(),
                acc_tys
                    .clone()
                    .into_list_parts()
                    .map(|sp| match sp {
                        SeqPart::Item(t) => t.as_runtime().unwrap(),
                        _ => unreachable!("Expected SeqPart::Item"),
                    })
                    .collect(),
            ))
        },
    );

    for op_def in BArrayOpDef::iter() {
        if op_def == BArrayOpDef::discard_empty {
            // discard_empty handled separately below.
            continue;
        }

        lw.replace_parametrized_op(
            borrow_array::EXTENSION.get_op(&op_def.opdef_id()).unwrap(),
            {
                move |args| {
                    // Since discard_empty is skipped, it is safe to assume [size, elem_ty].
                    let [size, elem_ty] = args else {
                        unreachable!()
                    };
                    let size = size.as_nat().unwrap();
                    let elem_ty = elem_ty.as_runtime().unwrap();

                    let node = match op_def {
                        BArrayOpDef::new_array => new_array_dest(size, elem_ty),
                        BArrayOpDef::get => get_dest(size, elem_ty),
                        BArrayOpDef::set => set_dest(size, elem_ty),
                        BArrayOpDef::swap => NodeTemplate::SingleOp(
                            ArrayOpDef::swap
                                .to_concrete(option_type(elem_ty).into(), size)
                                .into(),
                        ),
                        BArrayOpDef::pop_left => pop_dest(size, elem_ty, true),
                        BArrayOpDef::pop_right => pop_dest(size, elem_ty, false),
                        BArrayOpDef::unpack => unpack_dest(size, elem_ty),
                        _ => panic!("Unsupported BArrayOpDef variant {:?}", op_def),
                    };

                    Some(node)
                }
            },
        );
    }

    lw.replace_parametrized_op(
        borrow_array::EXTENSION
            .get_op(&BArrayOpDef::discard_empty.opdef_id())
            .unwrap(),
        move |args| {
            let [elem_ty] = args else { unreachable!() };
            Some(NodeTemplate::SingleOp(
                ArrayOpDef::discard_empty
                    .to_concrete(option_type(elem_ty.as_runtime().unwrap()).into(), 0)
                    .into(),
            ))
        },
    );

    lw.replace_parametrized_op(
        borrow_array::EXTENSION.get_op("to_array").unwrap(),
        move |args| {
            let [size, elem_ty] = args else {
                unreachable!()
            };
            Some(to_array_dest(
                size.as_nat().unwrap(),
                elem_ty.as_runtime().unwrap().clone(),
            ))
        },
    );

    lw.replace_parametrized_op(
        borrow_array::EXTENSION.get_op("from_array").unwrap(),
        move |args| {
            let [size, elem_ty] = args else {
                unreachable!()
            };
            Some(from_array_dest(
                size.as_nat().unwrap(),
                elem_ty.as_runtime().unwrap().clone(),
            ))
        },
    );

    lw
}

#[cfg(test)]
mod tests {
    use hugr::{
        builder::HugrBuilder,
        extension::prelude::{bool_t, qb_t, ConstUsize},
        std_extensions::collections::{
            array::op_builder::build_all_borrow_array_ops,
            borrow_array::{
                borrow_array_type, BArrayFromArray, BArrayOpBuilder, BArrayRepeat, BArrayScan,
                BArrayToArray,
            },
        },
        types::Signature,
        HugrView,
    };

    use super::*;

    #[test]
    fn test_array_type() {
        let size = 4;
        let elem_ty = qb_t();
        let ba_ty = borrow_array_type(size, elem_ty.clone());
        let dfb = DFGBuilder::new(inout_sig(vec![ba_ty.clone()], vec![ba_ty.clone()])).unwrap();
        let [ba] = dfb.input_wires_arr();
        let mut h = dfb.finish_hugr_with_outputs([ba]).unwrap();

        let pass = ReplaceBorrowArrayPass;
        pass.run(&mut h).unwrap();
        h.validate().unwrap();

        let expected_ty = array_type(size, option_type(elem_ty).into());
        let sig = h.signature(h.entrypoint()).unwrap();
        assert_eq!(sig.input(), &TypeRow::from(vec![expected_ty.clone()]));
        assert_eq!(sig.output(), &TypeRow::from(vec![expected_ty]));
    }

    #[test]
    fn test_const_array() {
        let size = 2;
        let elem_ty = usize_t();
        let ba_ty = borrow_array_type(size, elem_ty.clone());

        let ba_value = BArrayValue::new(
            usize_t(),
            vec![ConstUsize::new(0).into(), ConstUsize::new(0).into()],
        );

        let mut builder = DFGBuilder::new(Signature::new(vec![], vec![ba_ty.clone()])).unwrap();
        let arr = builder.add_load_value(ba_value);
        let mut h = builder.finish_hugr_with_outputs([arr]).unwrap();

        let pass = ReplaceBorrowArrayPass;
        pass.run(&mut h).unwrap();
        h.validate().unwrap();

        let expected_elem_ty = option_type(elem_ty.clone());
        let expected_ty = array_type(size, expected_elem_ty.clone().into());

        let sig = h.signature(h.entrypoint()).unwrap();
        assert_eq!(sig.output(), &TypeRow::from(vec![expected_ty.clone()]));
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
        let mut builder =
            DFGBuilder::new(Signature::new(vec![arr_ty.clone()], vec![qb_t()])).unwrap();
        let idx = builder.add_load_value(ConstUsize::new(0));
        let [arr] = builder.input_wires_arr();
        let (el, arr_with_borrowed) = builder
            .add_borrow_array_borrow(elem_ty.clone(), size, arr, idx)
            .unwrap();
        builder
            .add_discard_all_borrowed(elem_ty, size, arr_with_borrowed)
            .unwrap();
        let mut h = builder.finish_hugr_with_outputs([el]).unwrap();

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

    #[test]
    fn test_build_all_array_ops() {
        let sig = Signature::new_endo(Type::EMPTY_TYPEROW);
        let builder = DFGBuilder::new(sig).unwrap();
        let mut h = build_all_borrow_array_ops(builder).finish_hugr().unwrap();

        let pass = ReplaceBorrowArrayPass;
        pass.run(&mut h).unwrap();
        h.validate().unwrap();
    }

    #[test]
    fn test_clone_op() {
        let size = 3;
        let elem_ty = usize_t();
        let arr_ty = borrow_array_type(size, elem_ty.clone());
        let mut builder = DFGBuilder::new(Signature::new(
            vec![arr_ty.clone()],
            vec![arr_ty.clone(), arr_ty.clone()],
        ))
        .unwrap();
        let [arr] = builder.input_wires_arr();
        let (arr1, arr2) = builder
            .add_borrow_array_clone(elem_ty.clone(), size, arr)
            .unwrap();
        let mut h = builder.finish_hugr_with_outputs([arr1, arr2]).unwrap();

        let pass = ReplaceBorrowArrayPass;
        pass.run(&mut h).unwrap();
        h.validate().unwrap();
    }

    #[test]
    fn test_discard_op() {
        let size = 2;
        let elem_ty = usize_t();
        let arr_ty = borrow_array_type(size, elem_ty.clone());
        let mut builder = DFGBuilder::new(Signature::new(vec![arr_ty.clone()], vec![])).unwrap();
        let [arr] = builder.input_wires_arr();
        builder
            .add_borrow_array_discard(elem_ty.clone(), size, arr)
            .unwrap();
        let mut h = builder.finish_hugr_with_outputs([]).unwrap();

        let pass = ReplaceBorrowArrayPass;
        pass.run(&mut h).unwrap();
        h.validate().unwrap();
    }

    #[test]
    fn test_repeat_op() {
        let size = 4;
        let elem_ty = usize_t();
        let arr_ty = borrow_array_type(size, elem_ty.clone());
        let mut builder = DFGBuilder::new(Signature::new(vec![], vec![arr_ty.clone()])).unwrap();

        let mut module = builder.module_root_builder();
        let mut f = module
            .define_function("f", Signature::new(vec![], vec![elem_ty.clone()]))
            .unwrap();
        let out = f.add_load_value(ConstUsize::new(42));
        let func = f.finish_with_outputs([out]).unwrap();

        let func_input = builder.load_func(func.handle(), &[]).unwrap();

        let arr = builder
            .add_dataflow_op(BArrayRepeat::new(elem_ty.clone(), size), [func_input])
            .unwrap()
            .out_wire(0);
        let mut h = builder.finish_hugr_with_outputs([arr]).unwrap();

        let pass = ReplaceBorrowArrayPass;
        pass.run(&mut h).unwrap();
        h.validate().unwrap();
    }

    #[test]
    fn test_scan_op() {
        let size = 3;
        let src_ty = qb_t();
        let tgt_ty = bool_t();
        let arr_ty = borrow_array_type(size, src_ty.clone());

        let mut builder = DFGBuilder::new(Signature::new(
            vec![arr_ty.clone()],
            vec![borrow_array_type(size, tgt_ty.clone())],
        ))
        .unwrap();

        let mut module = builder.module_root_builder();
        let mut f = module
            .define_function(
                "map_to_bool",
                Signature::new(vec![src_ty.clone()], vec![tgt_ty.clone()]),
            )
            .unwrap();
        let out = f.add_load_value(Value::true_val());
        let func = f.finish_with_outputs([out]).unwrap();

        let [arr] = builder.input_wires_arr();
        let func_input = builder.load_func(func.handle(), &[]).unwrap();

        let scan_op = BArrayScan::new(src_ty.clone(), tgt_ty.clone(), vec![], size);
        let [out_arr] = builder
            .add_dataflow_op(scan_op, [arr, func_input])
            .unwrap()
            .outputs_arr();

        let mut h = builder.finish_hugr_with_outputs([out_arr]).unwrap();

        let pass = ReplaceBorrowArrayPass;
        pass.run(&mut h).unwrap();
        h.validate().unwrap();
    }

    #[test]
    fn test_roundtrip_from_and_to_array() {
        let size = 2;
        let elem_ty = qb_t();
        let standard_arr_ty = array_type(size, elem_ty.clone().into());

        let mut builder = DFGBuilder::new(Signature::new(
            vec![standard_arr_ty.clone()],
            vec![standard_arr_ty.clone()],
        ))
        .unwrap();
        let [input_arr] = builder.input_wires_arr();

        let borrowed = builder
            .add_dataflow_op(BArrayFromArray::new(elem_ty.clone(), size), [input_arr])
            .unwrap()
            .out_wire(0);

        let round_tripped = builder
            .add_dataflow_op(BArrayToArray::new(elem_ty.clone(), size), [borrowed])
            .unwrap()
            .out_wire(0);

        let mut h = builder.finish_hugr_with_outputs([round_tripped]).unwrap();

        let pass = ReplaceBorrowArrayPass;
        pass.run(&mut h).unwrap();
        h.validate().unwrap();

        let sig = h.signature(h.entrypoint()).unwrap();
        assert_eq!(sig.input(), &TypeRow::from(vec![standard_arr_ty.clone()]));
        assert_eq!(sig.output(), &TypeRow::from(vec![standard_arr_ty]));
    }
}
