//! Array codegen utilities.

use anyhow::Result;
use hugr::llvm::inkwell;
use inkwell::builder::{Builder, BuilderError};
use inkwell::context::Context;
use inkwell::types::{IntType, PointerType, StructType};
use inkwell::values::{ArrayValue, IntValue, PointerValue, StructValue};
use inkwell::AddressSpace;

/// Helper function to allocate an array on the stack.
///
/// Returns two pointers: The first one is a pointer to the first element of the
/// array (i.e. it is of type `array.get_element_type().ptr_type()`) whereas the
/// second one points to the whole array value, i.e. it is of type `array.ptr_type()`.
// Note: copied from
// https://github.com/CQCL/hugr/blob/bf3889fa206fbb5a22a5ae4b9ea5f8cc0468b4b7/hugr-llvm/src/extension/collections/array.rs#L186
pub fn build_array_alloca<'c>(
    builder: &Builder<'c>,
    array: ArrayValue<'c>,
) -> Result<(PointerValue<'c>, PointerValue<'c>), BuilderError> {
    let array_ty = array.get_type();
    let array_len: IntValue<'c> = {
        let ctx = builder.get_insert_block().unwrap().get_context();
        ctx.i32_type().const_int(u64::from(array_ty.len()), false)
    };
    let ptr = builder.build_array_alloca(array_ty.get_element_type(), array_len, "")?;
    let array_ptr = builder
        .build_bit_cast(ptr, array_ty.ptr_type(AddressSpace::default()), "")?
        .into_pointer_value();
    builder.build_store(array_ptr, array)?;
    Result::Ok((ptr, array_ptr))
}

/// Helper function to load an array from a pointer.
pub fn build_int_array_load<'c>(
    builder: &Builder<'c>,
    array_ptr: PointerValue<'c>,
    elem_type: IntType<'c>,
    length: u32,
) -> Result<ArrayValue<'c>, BuilderError> {
    let ptr = builder
        .build_bit_cast(
            array_ptr,
            elem_type
                .array_type(length)
                .ptr_type(AddressSpace::default()),
            "",
        )?
        .into_pointer_value();
    let array = builder.build_load(ptr, "")?.into_array_value();
    Result::Ok(array)
}

/// Enum representing the element types of a dense array.
#[allow(missing_docs)]
pub enum ElemType {
    Int,
    Uint,
    Float,
    Bool,
}

/// Helper function to create a dense array struct type.
///
/// The struct contains four fields:
///   (1) the size along the first data dimension
///   (2) the size along the second data dimension
///   (3) the pointer to the first element of the primary data in memory
///   (4) the pointer to the first element of the auxiliary sparsity flags
///       in memory
///
/// The fourth field points to an array of masking data of the same size as the
/// primary data in memory and contains boolean values to indicate the presence
/// of data in the primary array. Dense arrays have mask values of all zeros.
pub fn struct_1d_arr_t<'a>(ctx: &'a Context, data_type: &'a ElemType) -> StructType<'a> {
    let data_ptr_t = match data_type {
        ElemType::Int | ElemType::Uint => ctx.i64_type().ptr_type(AddressSpace::default()),
        ElemType::Float => ctx.f64_type().ptr_type(AddressSpace::default()),
        ElemType::Bool => ctx.bool_type().ptr_type(AddressSpace::default()),
    };
    ctx.struct_type(
        &[
            ctx.i32_type().into(), // x
            ctx.i32_type().into(), // y
            data_ptr_t.into(),     /* pointer to first array
                                    * elem */
            ctx.bool_type().ptr_type(AddressSpace::default()).into(), // pointer to first mask elem
        ],
        true,
    )
}

/// Helper function to create a `PointerType` to a dense array.
pub fn struct_1d_arr_ptr_t<'a>(ctx: &'a Context, data_type: &'a ElemType) -> PointerType<'a> {
    struct_1d_arr_t(ctx, data_type).ptr_type(AddressSpace::default())
}

/// Helper function to allocate and initialize a dense array struct on the stack.
///
/// Returns a `PointerVal` to the struct and the `StructVal` itself. All of
/// the mask values are initialized to 0.
pub fn struct_1d_arr_alloc<'a>(
    ctx: &'a Context,
    builder: &Builder<'a>,
    length: u32,
    data_type: &'a ElemType,
    array_ptr: PointerValue<'a>,
) -> Result<(PointerValue<'a>, StructValue<'a>), BuilderError> {
    let out_arr_type = struct_1d_arr_t(ctx, data_type);
    let out_arr_ptr = builder.build_alloca(out_arr_type, "out_arr_alloca")?;

    let x_field = builder.build_struct_gep(out_arr_ptr, 0, "x_ptr")?;
    let y_field = builder.build_struct_gep(out_arr_ptr, 1, "y_ptr")?;
    let arr_field = builder.build_struct_gep(out_arr_ptr, 2, "arr_ptr")?;
    let mask_field = builder.build_struct_gep(out_arr_ptr, 3, "mask_ptr")?;

    let x_val = ctx.i32_type().const_int(length.into(), false);
    let y_val = ctx.i32_type().const_int(1, false);
    let bit_cast_type = match data_type {
        ElemType::Bool => ctx.bool_type().ptr_type(AddressSpace::default()),
        ElemType::Int | ElemType::Uint => ctx.i64_type().ptr_type(AddressSpace::default()),
        ElemType::Float => ctx.f64_type().ptr_type(AddressSpace::default()),
    };
    let casted_arr_ptr = builder
        .build_bit_cast(array_ptr, bit_cast_type, "")?
        .into_pointer_value();
    let (mask_ptr, _) = build_array_alloca(
        builder,
        ctx.bool_type().const_array(
            vec![ctx.bool_type().const_int(0, false); length.try_into().unwrap()].as_slice(),
        ),
    )?;

    builder.build_store(x_field, x_val)?;
    builder.build_store(y_field, y_val)?;
    builder.build_store(arr_field, casted_arr_ptr)?;
    builder.build_store(mask_field, mask_ptr)?;

    let out_arr = builder.build_load(out_arr_ptr, "")?.into_struct_value();

    Result::Ok((out_arr_ptr, out_arr))
}
