//! Array codegen utilities.

// TODO move to hugr-llvm crate
// https://github.com/CQCL/tket2/issues/899
use anyhow::Result;
use hugr::llvm::extension::collections::array::decompose_array_fat_pointer;
use hugr::llvm::extension::collections::{array, stack_array};
use hugr::llvm::inkwell::values::BasicValueEnum;
use hugr::llvm::{inkwell, CodegenExtension};
use inkwell::builder::{Builder, BuilderError};
use inkwell::context::Context;
use inkwell::types::{IntType, PointerType, StructType};
use inkwell::values::{ArrayValue, IntValue, PointerValue, StructValue};
use inkwell::AddressSpace;

/// Specifies different array lowering strategies.
///
/// See [DEFAULT_STACK_ARRAY_LOWERING] and [DEFAULT_HEAP_ARRAY_LOWERING] for the default
/// array lowerings implementing this trait.
pub trait ArrayLowering {
    /// The [CodegenExtension] specifying the array lowering.
    fn codegen_extension(&self) -> impl CodegenExtension;

    /// Turns an array value in the given lowering into a pointer to the first array element.
    fn array_to_ptr<'c>(
        &self,
        builder: &Builder<'c>,
        val: BasicValueEnum<'c>,
    ) -> Result<PointerValue<'c>>;
}

/// Array lowering via the stack as implemented in [stack_array].
#[derive(Clone)]
#[allow(deprecated)]
pub struct StackArrayLowering<ACG: stack_array::ArrayCodegen>(ACG);

/// The default stack array lowering strategy using [stack_array::DefaultArrayCodegen].
#[allow(deprecated)]
pub const DEFAULT_STACK_ARRAY_LOWERING: StackArrayLowering<stack_array::DefaultArrayCodegen> =
    StackArrayLowering(stack_array::DefaultArrayCodegen);

#[allow(deprecated)]
impl<ACG: stack_array::ArrayCodegen> StackArrayLowering<ACG> {
    /// Creates a new [StackArrayLowering].
    pub const fn new(array_codegen: ACG) -> Self {
        Self(array_codegen)
    }
}

#[allow(deprecated)]
impl<ACG: stack_array::ArrayCodegen + Clone> ArrayLowering for StackArrayLowering<ACG> {
    fn codegen_extension(&self) -> impl CodegenExtension {
        stack_array::ArrayCodegenExtension::new(self.0.clone())
    }

    fn array_to_ptr<'c>(
        &self,
        builder: &Builder<'c>,
        val: BasicValueEnum<'c>,
    ) -> Result<PointerValue<'c>> {
        let (elem_ptr, _) = build_array_alloca(builder, val.into_array_value())?;
        Ok(elem_ptr)
    }
}

/// Array lowering via a heap as implemented in [array].
#[derive(Clone)]
pub struct HeapArrayLowering<ACG: array::ArrayCodegen>(ACG);

/// The default heap array lowering strategy using [array::DefaultArrayCodegen].
pub const DEFAULT_HEAP_ARRAY_LOWERING: HeapArrayLowering<array::DefaultArrayCodegen> =
    HeapArrayLowering(array::DefaultArrayCodegen);

impl<ACG: array::ArrayCodegen> HeapArrayLowering<ACG> {
    /// Creates a new [HeapArrayLowering].
    pub const fn new(array_codegen: ACG) -> Self {
        Self(array_codegen)
    }
}

impl<ACG: array::ArrayCodegen + Clone> ArrayLowering for HeapArrayLowering<ACG> {
    fn codegen_extension(&self) -> impl CodegenExtension {
        array::ArrayCodegenExtension::new(self.0.clone())
    }

    fn array_to_ptr<'c>(
        &self,
        builder: &Builder<'c>,
        val: BasicValueEnum<'c>,
    ) -> Result<PointerValue<'c>> {
        let (array_ptr, offset) = decompose_array_fat_pointer(builder, val)?;
        let elem_ptr = unsafe { builder.build_in_bounds_gep(array_ptr, &[offset], "")? };
        Ok(elem_ptr)
    }
}

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

#[cfg(test)]
mod tests {
    use super::*;
    use hugr::llvm::inkwell::context::Context;

    /// Test that build_array_alloca properly allocates an array.
    #[test]
    fn test_build_array_alloca() {
        let context = Context::create();
        let module = context.create_module("test_module");
        let builder = context.create_builder();

        make_bb(&context, &module, &builder);

        let (ptr, array_ptr) =
            build_array(&context, &builder).expect("Array allocation should succeed");

        assert!(!ptr.is_null(), "Element pointer should not be null");
        assert!(!array_ptr.is_null(), "Array pointer should not be null");

        builder
            .build_return(None)
            .expect("Should be able to build return");

        // Verify the generated code is valid
        assert!(module.verify().is_ok(), "Module verification failed");
    }

    /// Helper function to create a basic block for testing.
    fn make_bb<'c>(
        context: &'c Context,
        module: &inkwell::module::Module<'c>,
        builder: &Builder<'c>,
    ) {
        let function_type = context.void_type().fn_type(&[], false);
        let function = module.add_function("test_function", function_type, None);
        let basic_block = context.append_basic_block(function, "entry");
        builder.position_at_end(basic_block);
    }

    fn build_array<'c>(
        context: &'c Context,
        builder: &Builder<'c>,
    ) -> Result<(PointerValue<'c>, PointerValue<'c>), BuilderError> {
        // Create test array
        let i32_type = context.i32_type();
        let array =
            i32_type.const_array(&[i32_type.const_int(1, false), i32_type.const_int(2, false)]);

        build_array_alloca(builder, array)
    }

    /// Test that build_int_array_load properly loads an array.
    #[test]
    fn test_build_int_array_load() {
        let context = Context::create();
        let module = context.create_module("test_module");
        let builder = context.create_builder();

        make_bb(&context, &module, &builder);

        let (array_ptr, _) =
            build_array(&context, &builder).expect("Array allocation should succeed");
        let i32_type = context.i32_type();
        let array_length = 2;
        let loaded_array = build_int_array_load(&builder, array_ptr, i32_type, array_length)
            .expect("Array load should succeed");

        assert_eq!(loaded_array.get_type().len(), array_length,);

        builder.build_return(None).unwrap();

        // Verify the generated code is valid
        assert!(module.verify().is_ok(), "Module verification failed");
    }

    /// Test that struct_1d_arr_t creates the correct structure type.
    #[test]
    fn test_struct_1d_arr_t() {
        let context = Context::create();

        // Test for each element type
        let int_struct = struct_1d_arr_t(&context, &ElemType::Int);
        let uint_struct = struct_1d_arr_t(&context, &ElemType::Uint);
        let float_struct = struct_1d_arr_t(&context, &ElemType::Float);
        let bool_struct = struct_1d_arr_t(&context, &ElemType::Bool);

        let structs = [int_struct, uint_struct, float_struct, bool_struct];

        for s in &structs {
            // All structs should have 4 fields
            assert_eq!(s.get_field_types().len(), 4);

            // Check the field types (first two fields should be i32 for all structs)
            assert!(s.get_field_types()[0].is_int_type());
            assert!(s.get_field_types()[1].is_int_type());

            // Third field should be a pointer to the corresponding data type
            assert!(s.get_field_types()[2].is_pointer_type());

            // Fourth field should be a pointer to bool type for all structs
            assert!(s.get_field_types()[3].is_pointer_type());
        }
    }

    /// Test that struct_1d_arr_ptr_t returns the correct pointer type.
    #[test]
    fn test_struct_1d_arr_ptr_t() {
        let context = Context::create();

        // Test for each element type
        let int_ptr = struct_1d_arr_ptr_t(&context, &ElemType::Int);
        let uint_ptr = struct_1d_arr_ptr_t(&context, &ElemType::Uint);
        let float_ptr = struct_1d_arr_ptr_t(&context, &ElemType::Float);
        let bool_ptr = struct_1d_arr_ptr_t(&context, &ElemType::Bool);

        // Test that all element types return struct pointer types
        for ptr in [int_ptr, uint_ptr, float_ptr, bool_ptr] {
            assert!(ptr.get_element_type().is_struct_type());
        }
    }

    /// Test that struct_1d_arr_alloc properly allocates and initializes a dense array struct.
    #[test]
    fn test_struct_1d_arr_alloc() {
        let context = Context::create();
        let module = context.create_module("test_module");
        let builder = context.create_builder();

        make_bb(&context, &module, &builder);

        let (array_ptr, _) = build_array(&context, &builder).unwrap();
        // Test the function with different element types
        let elem_types = [ElemType::Int, ElemType::Float, ElemType::Bool];

        for elem_type in elem_types.iter() {
            let (struct_ptr, _) =
                struct_1d_arr_alloc(&context, &builder, 2, elem_type, array_ptr).unwrap();

            assert!(!struct_ptr.is_null(), "Struct pointer should not be null");
        }

        builder
            .build_return(None)
            .expect("Should be able to build return");

        // Verify the generated code is valid
        assert!(module.verify().is_ok(), "Module verification failed");
    }
}
