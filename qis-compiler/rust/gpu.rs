//! GPU function API lowering - API version 0.1.0.
//!
//! This module lowers the "tket.gpu" extension to the following external
//! calls:
//!
//! - Validation
//!   ```rust
//!   gpu_validate_api(major: u64, minor: u64, patch: u64) -> bool
//!   ```
//!
//!   Validates that this API is compatible with the GPU library linked
//!   with the final binary.
//!
//!   The present version is 0.1.0.
//!
//!   We opt to allow the linked library to perform validation, as it
//!   may support multiple API versions.
//!
//! - Construction
//!   ```rust
//!   gpu_init(_reserved: u64, gpu_ref_out: *mut u64) -> bool
//!   ```
//!
//!   The reserved argument must be 0 for now.
//!   On success, writes to handle_out a handle to pass to GPU calls.
//!   On failure, returns false.
//!
//! - Destruction
//!   ```rust
//!   gpu_discard(gpu_ref: u64) -> bool
//!   ```
//!
//!   Attempts to clean up the GPU context referenced by gpu_ref.
//!   This can fail - for example, if gpu_ref is invalid, or if
//!   cleanup validation fails. Returns true on success, false on failure.
//!
//! - Error Retrieval
//!   ```rust
//!   gpu_get_error() -> *const i8
//!   ```
//!
//!   Returns a pointer to a null-terminated error string, or nullptr
//!   if no error has been recorded.
//!
//! - Function extraction
//!   ```rust
//!   gpu_get_function_id(name: *const i8, id_out: *mut u64) -> bool
//!   ```
//!
//!   Provides a function id for the given function name through id_out on success,
//!   returning true. If the function name is not found, or gpu_ref is invalid,
//!   returns false.
//!
//! - Function invocation
//!   ```rust
//!   gpu_call(
//!     handle: u64,
//!     function_id: u64,
//!     blob_size: u64,
//!     blob: *const i8,
//!     signature: *const i8
//!   ) -> bool
//!   ```
//!
//!   The GPU api is specified by the user, and arguments are passed
//!   in as a binary blob. To provide validation on the library side,
//!   we provide the blob size and a string representing the function
//!   signature that the user expects. The latter is in the form of
//!   e.g.:
//!   - 'iifb:v' for (i64, u64, f64, bool) -> void
//!   - 'if:i' for (i64, f64) -> i64
//!
//!   where types are encoded as:
//!   - i <=> i64 or u64
//!   - f <=> f64
//!   - b <=> bool
//!
//!  - Result retrieval
//!    ```rust
//!    gpu_get_result_64bits(gpu_ref: u64, out_result: *mut i64) -> bool
//!    ```
//!
//!    Writes the next result in the result queue to out_result,
//!    returning true on success or false on failure. The result
//!    may be in the form of i64 or f64; we handle any necessary
//!    casting within this lowering.
//!

use crate::selene_specific;
use anyhow::{Result, bail};
use hugr::llvm::{CodegenExtension, CodegenExtsBuilder, inkwell};
use hugr::{HugrView, Node};
use inkwell::AddressSpace;
use inkwell::types::{BasicType as _, BasicTypeEnum};
use inkwell::values::{BasicMetadataValueEnum, BasicValue, BasicValueEnum};
use std::cmp::Ordering;
use strum::IntoEnumIterator;
use tket::hugr::extension::ExtensionId;
use tket::hugr::extension::simple_op::{MakeExtensionOp, MakeOpDef};
use tket::hugr::llvm::emit::{EmitFuncContext, EmitOpArgs};
use tket::hugr::ops::ExtensionOp;
use tket::hugr::types::TypeRow;
use tket::hugr::{self};
use tket_qsystem::extension::classical_compute::{self, ComputeOp};
use tket_qsystem::extension::gpu::{self, ConstGpuModule, GpuExtension, GpuOpDef};

const API_MAJOR: u64 = 0;
const API_MINOR: u64 = 1;
const API_PATCH: u64 = 0;

#[derive(Debug, Clone)]
pub struct GpuCodegen;

impl GpuCodegen {
    /// Emit the GetContext operation. This initializes a GPU context
    /// using the external `gpu_init` function, and returns a pair
    /// representing success and the GPU reference itself.
    ///
    /// Although the GpuExtension's GetContext op returns an Option type,
    /// and thus handles failure gracefully, it lowers to a QIS panic, which
    /// is restricted in length and may not be able to convey a generic error
    /// message from the GPU library - which might contain verbose information
    /// pertinent to the failure.
    ///
    /// As such we utilise the verify_gpu_call helper to emit a panic directly
    /// via selene's panic_str function, which can handle c strings.
    fn emit_get_context<'c, H: HugrView<Node = Node>>(
        &self,
        ctx: &EmitFuncContext<'c, '_, H>,
        op: EmitOpArgs<'c, '_, ExtensionOp, H>,
    ) -> Result<()> {
        let [index] = &op.inputs[..] else {
            bail!("GetContext operation requires exactly one input");
        };
        let index = {
            let index = index.into_int_value();
            let width = index.get_type().get_bit_width();
            match width.cmp(&64) {
                Ordering::Greater => ctx.builder().build_int_truncate(
                    index,
                    ctx.iw_context().i64_type(),
                    "context_index_trunc",
                )?,
                Ordering::Less => ctx.builder().build_int_z_extend(
                    index,
                    ctx.iw_context().i64_type(),
                    "context_index_zext",
                )?,
                Ordering::Equal => index,
            }
        };
        let iwc = ctx.iw_context();
        let builder = ctx.builder();

        let handle_ptr = builder.build_alloca(iwc.i64_type(), "handle_ptr")?;

        let get_gpu_init = ctx.get_extern_func(
            "gpu_init",
            iwc.i8_type().fn_type(
                &[
                    iwc.i64_type().into(),
                    iwc.i64_type().ptr_type(AddressSpace::default()).into(),
                ],
                false,
            ),
        )?;

        // `gpu_init` returns false on failure. Otherwise, it provides
        // in handle_ptr a handle to the GPU context, which is
        // used for subsequent GPU calls.
        let success = builder
            .build_call(
                get_gpu_init,
                &[index.into(), handle_ptr.into()],
                "gpu_ref_call",
            )?
            .try_as_basic_value()
            .unwrap_left()
            .into_int_value();
        verify_gpu_call(ctx, success, "gpu_init")?;

        let gpu_ref = builder.build_load(handle_ptr, "gpu_ref")?.into_int_value();

        let struct_ty = iwc.struct_type(&[iwc.bool_type().into(), iwc.i64_type().into()], false);
        // Failure has already been handled by verify_gpu_call, so always
        // return success.
        let pair = builder.build_insert_value(
            struct_ty.get_undef(),
            iwc.bool_type().const_int(1, false),
            0,
            "status",
        )?;
        let pair =
            builder.build_insert_value(pair.into_struct_value(), gpu_ref, 1, "gpu_ref_result")?;
        op.outputs
            .finish(ctx.builder(), [pair.as_basic_value_enum()])
    }

    /// Emit the DisposeContext operation. This cleans up the GPU context
    /// using the external `gpu_discard` function.
    ///
    /// On failure, this emits a panic with the error message
    /// from the GPU library.
    fn emit_dispose_context<'c, H: HugrView<Node = Node>>(
        &self,
        ctx: &EmitFuncContext<'c, '_, H>,
        op: EmitOpArgs<'c, '_, ExtensionOp, H>,
    ) -> Result<()> {
        let [gpu_ref] = &op.inputs[..] else {
            bail!("DisposeContext operation requires exactly one input");
        };
        let gpu_ref = gpu_ref.into_int_value();
        let iwc = ctx.iw_context();

        let gpu_discard = ctx.get_extern_func(
            "gpu_discard",
            iwc.i8_type().fn_type(&[iwc.i64_type().into()], false),
        )?;

        let success = ctx
            .builder()
            .build_call(gpu_discard, &[gpu_ref.into()], "")?
            .try_as_basic_value()
            .unwrap_left()
            .into_int_value();
        verify_gpu_call(ctx, success, "gpu_discard")?;

        op.outputs.finish(ctx.builder(), [])
    }
    fn emit_lookup_by_name<'c, H: HugrView<Node = Node>>(
        &self,
        ctx: &EmitFuncContext<'c, '_, H>,
        op: EmitOpArgs<'c, '_, ExtensionOp, H>,
        name: String,
    ) -> Result<()> {
        let iwc = ctx.iw_context();
        let global_name = format!("gpu_function_id_{}", name);
        let func_id = match ctx.get_current_module().get_global(&global_name) {
            None => {
                let stored =
                    ctx.get_current_module()
                        .add_global(iwc.i64_type(), None, &global_name);
                stored.set_initializer(&iwc.i64_type().const_zero());

                let gpu_get_function_id = ctx.get_extern_func(
                    "gpu_get_function_id",
                    iwc.i8_type().fn_type(
                        &[
                            iwc.i8_type().ptr_type(AddressSpace::default()).into(),
                            iwc.i64_type().ptr_type(AddressSpace::default()).into(),
                        ],
                        false,
                    ),
                )?;
                let name_const = ctx
                    .builder()
                    .build_global_string_ptr(&name, "function_name")?;
                let function_id_ptr = ctx
                    .builder()
                    .build_alloca(iwc.i64_type(), "function_id_ptr")?;
                let success = ctx
                    .builder()
                    .build_call(
                        gpu_get_function_id,
                        &[name_const.as_pointer_value().into(), function_id_ptr.into()],
                        "function_id_call",
                    )?
                    .try_as_basic_value()
                    .unwrap_left()
                    .into_int_value();
                verify_gpu_call(ctx, success, "gpu_get_function_id")?;
                let func_id = ctx
                    .builder()
                    .build_load(function_id_ptr, "function_id")?
                    .into_int_value();
                ctx.builder()
                    .build_store(stored.as_pointer_value(), func_id)?;
                func_id
            }
            Some(stored) => ctx
                .builder()
                .build_load(stored.as_pointer_value(), "function_id")?
                .into_int_value(),
        };
        op.outputs.finish(ctx.builder(), [func_id.into()])
    }

    fn emit_lookup_by_id<'c, H: HugrView<Node = Node>>(
        &self,
        ctx: &EmitFuncContext<'c, '_, H>,
        op: EmitOpArgs<'c, '_, ExtensionOp, H>,
        id: u64,
    ) -> Result<()> {
        op.outputs.finish(
            ctx.builder(),
            [ctx.iw_context().i64_type().const_int(id, false).into()],
        )
    }

    /// Emit the Call operation. This invokes a GPU function using
    /// the external `gpu_call` function.
    ///
    /// Arguments are packed into a binary blob, and a signature
    /// string is generated to describe the types of the arguments
    /// and the expected output type, which may be used by the
    /// GPU library for validation purposes.
    ///
    /// Upon failure, this emits a panic with the error message
    /// from the GPU library.
    fn emit_gpu_call<'c, H: HugrView<Node = Node>>(
        &self,
        ctx: &EmitFuncContext<'c, '_, H>,
        op: EmitOpArgs<'c, '_, ExtensionOp, H>,
        outputs: TypeRow,
    ) -> Result<()> {
        let iwc = ctx.iw_context();

        let [gpu_ref, fn_id, fn_args @ ..] = &op.inputs[..] else {
            bail!("GPU call operation requires at least two inputs: context and function");
        };
        let gpu_ref = gpu_ref.into_int_value();
        let func = fn_id.into_int_value();

        let builder = ctx.builder();

        let gpu_call = ctx.get_extern_func(
            "gpu_call",
            iwc.i8_type().fn_type(
                &[
                    gpu_ref.get_type().into(),
                    func.get_type().into(),
                    iwc.i64_type().into(),
                    iwc.i8_type()
                        .ptr_type(inkwell::AddressSpace::default())
                        .into(),
                    iwc.i8_type()
                        .ptr_type(inkwell::AddressSpace::default())
                        .into(),
                ],
                false,
            ),
        )?;

        let mut args: Vec<BasicMetadataValueEnum> = vec![gpu_ref.into(), func.into()];

        let signature = generate_signature(ctx, fn_args, outputs)?;

        // create an i8 array of length blob_size
        let i64_zero = iwc.i64_type().const_zero();
        let (blob, blob_size) = pack_arguments(ctx, fn_args)?;
        let blob_ptr = unsafe {
            builder.build_in_bounds_gep(blob, &[i64_zero, i64_zero], "gpu_input_blob_ptr")?
        };

        // pass the blob size, blob pointer, and signature string to gpu_call as
        // arguments
        args.push(iwc.i64_type().const_int(blob_size as u64, false).into());
        args.push(blob_ptr.into());
        args.push(signature.as_pointer_value().into());

        let success = ctx
            .builder()
            .build_call(gpu_call, &args, "")?
            .try_as_basic_value()
            .unwrap_left()
            .into_int_value();
        verify_gpu_call(ctx, success, "gpu_call")?;
        op.outputs.finish(ctx.builder(), [gpu_ref.into()])?;
        Ok(())
    }

    fn emit_read_result<'c, H: HugrView<Node = Node>>(
        &self,
        ctx: &EmitFuncContext<'c, '_, H>,
        op: EmitOpArgs<'c, '_, ExtensionOp, H>,
        outputs: TypeRow,
    ) -> Result<()> {
        let [gpu_ref] = &op.inputs[..] else {
            bail!("ReadResult operation requires exactly one input");
        };
        let gpu_ref = gpu_ref.into_int_value();

        match &outputs[..] {
            [] => {
                op.outputs
                    .finish(ctx.builder(), [gpu_ref.as_basic_value_enum()])?;
            }
            [single_result] => {
                let iwc = ctx.iw_context();
                let i8_t = iwc.i8_type().as_basic_type_enum();
                let i64_t = iwc.i64_type().as_basic_type_enum();
                let f64_t = iwc.f64_type().as_basic_type_enum();
                let int_result_ptr_t = iwc.i64_type().ptr_type(AddressSpace::default());

                // Results can come in as ints or floats. We receive it as an int into int_result,
                // and reinterpret it as a float if required.
                let int_result_ptr = ctx.builder().build_alloca(iwc.i64_type(), "int_result")?;

                let read_func = ctx.get_extern_func(
                    "gpu_get_result_64bits",
                    i8_t.fn_type(
                        &[
                            /* gpu_ref    */ iwc.i64_type().into(),
                            /* out result */ int_result_ptr_t.into(),
                        ],
                        false,
                    ),
                )?;

                let call = ctx
                    .builder()
                    .build_call(
                        read_func,
                        &[gpu_ref.into(), int_result_ptr.into()],
                        "read_status",
                    )?
                    .try_as_basic_value()
                    .unwrap_left()
                    .into_int_value();
                // Check status and handle error if needed
                verify_gpu_call(ctx, call, "gpu_get_result_64bits")?;

                match ctx.llvm_type(single_result)? {
                    i if i == i64_t => {
                        // The result type is an integer, so we can just load
                        // it directly.
                        let int_result = ctx
                            .builder()
                            .build_load(int_result_ptr, "int_result")?
                            .into_int_value();
                        op.outputs.finish(
                            ctx.builder(),
                            [
                                gpu_ref.as_basic_value_enum(),
                                int_result.as_basic_value_enum(),
                            ],
                        )?;
                    }
                    f if f == f64_t => {
                        // The result type is a float, so we need to reinterpret
                        // the result pointer from a pointer to int to a pointer
                        // to float, then load it.
                        let float_result_ptr = ctx
                            .builder()
                            .build_bit_cast(
                                int_result_ptr,
                                iwc.f64_type().ptr_type(AddressSpace::default()),
                                "float_result_ptr",
                            )?
                            .into_pointer_value();

                        let float_result = ctx
                            .builder()
                            .build_load(float_result_ptr, "float_result")?
                            .into_float_value();
                        op.outputs.finish(
                            ctx.builder(),
                            [
                                gpu_ref.as_basic_value_enum(),
                                float_result.as_basic_value_enum(),
                            ],
                        )?;
                    }
                    _ => {
                        bail!(
                            "ReadResult operation with a single output expects either i64 or f64, got: {:?}",
                            ctx.llvm_type(single_result)?
                        );
                    }
                }
            }
            other => {
                bail!("ReadResult operation expects either zero or one, got: {other:?}")
            }
        }
        Ok(())
    }

    fn emit_op<'c, H: HugrView<Node = Node>>(
        &self,
        ctx: &EmitFuncContext<'c, '_, H>,
        op: EmitOpArgs<'c, '_, ExtensionOp, H>,
    ) -> Result<()> {
        match ComputeOp::<GpuExtension>::from_extension_op(&op.node())? {
            ComputeOp::<GpuExtension>::GetContext => self.emit_get_context(ctx, op),
            ComputeOp::<GpuExtension>::DisposeContext => self.emit_dispose_context(ctx, op),
            ComputeOp::<GpuExtension>::LookupById { id, .. } => self.emit_lookup_by_id(ctx, op, id),
            ComputeOp::<GpuExtension>::LookupByName { name, .. } => {
                self.emit_lookup_by_name(ctx, op, name)
            }
            ComputeOp::<GpuExtension>::Call { outputs, .. } => self.emit_gpu_call(ctx, op, outputs),
            ComputeOp::<GpuExtension>::ReadResult { outputs } => {
                self.emit_read_result(ctx, op, outputs)
            }
            ComputeOp::<GpuExtension>::_Unreachable(x, _) => match x {},
        }
    }

    fn add_gpu_extension<'a, H: HugrView<Node = Node> + 'a, T: MakeOpDef + IntoEnumIterator>(
        self,
        extension_id: ExtensionId,
        builder: CodegenExtsBuilder<'a, H>,
    ) -> CodegenExtsBuilder<'a, H>
    where
        Self: 'a + Sized,
    {
        builder
            .custom_type(
                (
                    extension_id.clone(),
                    classical_compute::CONTEXT_TYPE_NAME.to_owned(),
                ),
                |session, _hugr_type| Ok(session.iw_context().i64_type().into()),
            )
            .custom_type(
                (
                    extension_id.clone(),
                    classical_compute::FUNC_TYPE_NAME.to_owned(),
                ),
                |session, _hugr_type| Ok(session.iw_context().i64_type().into()),
            )
            .custom_type(
                (
                    extension_id.clone(),
                    classical_compute::MODULE_TYPE_NAME.to_owned(),
                ),
                |session, _hugr_type| Ok(session.iw_context().struct_type(&[], false).into()),
            )
            .custom_type(
                (extension_id, classical_compute::RESULT_TYPE_NAME.to_owned()),
                |session, _hugr_type| Ok(session.iw_context().i64_type().into()),
            )
            .simple_extension_op::<T>(move |context, args, _| self.emit_op(context, args))
            .custom_const::<ConstGpuModule>({
                move |ctx, _mod| {
                    emit_api_validation(ctx)?;
                    Ok(ctx.iw_context().const_struct(&[], false).into())
                }
            })
    }
}

impl CodegenExtension for GpuCodegen {
    fn add_extension<'a, H: HugrView<Node = Node> + 'a>(
        self,
        builder: CodegenExtsBuilder<'a, H>,
    ) -> CodegenExtsBuilder<'a, H> {
        self.add_gpu_extension::<'a, H, GpuOpDef>(gpu::EXTENSION_ID, builder)
    }
}

// Helpers

fn emit_api_validation<'c, H: HugrView<Node = Node>>(
    ctx: &EmitFuncContext<'c, '_, H>,
) -> Result<()> {
    let iwc = ctx.iw_context();
    let module = ctx.get_current_module();
    if module.get_global("gpu_validated").is_some() {
        // Already validated
        return Ok(());
    } else {
        module
            .add_global(iwc.i8_type(), None, "gpu_validated")
            .set_initializer(&iwc.i8_type().const_int(1, false));
    }
    let builder = ctx.builder();
    let major = iwc.i64_type().const_int(API_MAJOR, false);
    let minor = iwc.i64_type().const_int(API_MINOR, false);
    let patch = iwc.i64_type().const_int(API_PATCH, false);
    let validate_func = ctx.get_extern_func(
        "gpu_validate_api",
        iwc.i8_type().fn_type(
            &[
                iwc.i64_type().into(),
                iwc.i64_type().into(),
                iwc.i64_type().into(),
            ],
            false,
        ),
    )?;
    let success = builder
        .build_call(
            validate_func,
            &[major.into(), minor.into(), patch.into()],
            "validate_call",
        )?
        .try_as_basic_value()
        .unwrap_left()
        .into_int_value();
    verify_gpu_call(ctx, success, "gpu_validate_api")?;
    Ok(())
}

/// A helper method for handling errors from the external
/// library.
///
/// For GPU calls, we have no recovery mechanism built-in, so
/// we emit a panic with the error message from the library.
/// If no error message is available, we use a default message.
///
/// Utilises the selene panic_str function to emit the panic,
/// rather than the size-restricted QIS panic.
fn emit_panic_with_gpu_error<'c, H: HugrView<Node = Node>>(
    ctx: &EmitFuncContext<'c, '_, H>,
) -> Result<()> {
    let module = ctx.get_current_module();
    let iwc = ctx.iw_context();
    let builder = ctx.builder();
    let handle_error = match module.get_function("gpu_error_handler") {
        None => {
            let current_block = builder.get_insert_block().unwrap();
            let fn_type = iwc.void_type().fn_type(&[], false);
            let function = module.add_function("gpu_error_handler", fn_type, None);
            let noinline_id = inkwell::attributes::Attribute::get_named_enum_kind_id("noinline");
            let noinline = iwc.create_enum_attribute(noinline_id, 0);
            function.add_attribute(inkwell::attributes::AttributeLoc::Function, noinline);
            let entry = iwc.append_basic_block(function, "entry");
            builder.position_at_end(entry);
            // Try to get the error message from the GPU library.
            let gpu_get_error = ctx.get_extern_func(
                "gpu_get_error",
                iwc.i8_type()
                    .ptr_type(AddressSpace::default())
                    .fn_type(&[], false),
            )?;

            let error_message = ctx
                .builder()
                .build_call(gpu_get_error, &[], "error_message")?
                .try_as_basic_value()
                .unwrap_left()
                .into_pointer_value();

            // If it's null, replace with "No error message available"
            let error_message = ctx.builder().build_select(
                ctx.builder().build_int_compare(
                    inkwell::IntPredicate::EQ,
                    error_message,
                    iwc.i8_type().ptr_type(AddressSpace::default()).const_null(),
                    "is_null",
                )?,
                ctx.builder()
                    .build_global_string_ptr("No error message available", "no_gpu_error")?
                    .as_pointer_value(),
                error_message,
                "error_message_nonnull",
            )?;

            // Call panic_str with a generic error code and the message.
            ctx.builder()
                .build_call(
                    selene_specific::panic_str_fn(ctx)?,
                    &[
                        iwc.i32_type().const_int(70002, false).into(),
                        error_message.into(),
                    ],
                    "panic_str_call",
                )?
                .try_as_basic_value()
                .unwrap_right();
            builder.build_unreachable()?;
            builder.position_at_end(current_block);
            function
        }
        Some(f) => f,
    };
    builder
        .build_call(handle_error, &[], "gpu_error_handler_call")?
        .try_as_basic_value()
        .unwrap_right();
    Ok(())
}

/// A helper to check return codes and handle the branching
/// to `emit_panic_with_gpu_error` upon error.
/// If the return value is true, we continue as normal.
fn verify_gpu_call<'c, H: HugrView<Node = Node>>(
    ctx: &EmitFuncContext<'c, '_, H>,
    success_flag: inkwell::values::IntValue<'c>,
    op_name: &str,
) -> Result<()> {
    let module = ctx.get_current_module();
    let iwc = ctx.iw_context();
    let builder = ctx.builder();
    // Rather than emiting the validation code inline after each gpu call,
    // we define a response validation function and call into it after each
    // gpu call. Keeping it noinline keeps the resulting IR clean, but
    // we can always inline it later if required.
    let handle_error = match module.get_function("validate_gpu_response") {
        None => {
            let current_block = builder.get_insert_block().unwrap();
            let fn_type = iwc.void_type().fn_type(&[iwc.i8_type().into()], false);
            let function = module.add_function("validate_gpu_response", fn_type, None);
            let noinline_id = inkwell::attributes::Attribute::get_named_enum_kind_id("noinline");
            let noinline = iwc.create_enum_attribute(noinline_id, 0);
            function.add_attribute(inkwell::attributes::AttributeLoc::Function, noinline);
            let entry = iwc.append_basic_block(function, "entry");
            builder.position_at_end(entry);
            let ok_block = iwc.append_basic_block(function, "ok");
            let err_block = iwc.append_basic_block(function, "err");
            let success_bool = builder.build_int_compare(
                inkwell::IntPredicate::NE,
                function.get_first_param().unwrap().into_int_value(),
                iwc.i8_type().const_zero(),
                "success",
            )?;
            builder.build_conditional_branch(success_bool, ok_block, err_block)?;
            // on error, we panic
            builder.position_at_end(err_block);
            emit_panic_with_gpu_error(ctx)?;
            builder.build_unreachable()?;
            // on success, return
            builder.position_at_end(ok_block);
            builder.build_return(None)?;
            builder.position_at_end(current_block);
            function
        }
        Some(f) => f,
    };
    builder
        .build_call(
            handle_error,
            &[success_flag.into()],
            format!("{op_name}_handle_error").as_str(),
        )?
        .try_as_basic_value()
        .unwrap_right();
    Ok(())
}

/// A helper to generate a type signature string
/// for the GPU call, given a list of arguments.
///
/// The signature is of the form "arg_types:output_type"
/// where arg_types is a sequence of characters representing
/// the types of each argument, and output_type is
/// a single character representing the output type.
///
///  Types are encoded as:
///   i - i64 or u64
///   f - f64
///   b - bool
fn generate_signature<'c, H: HugrView<Node = Node>>(
    ctx: &EmitFuncContext<'c, '_, H>,
    fn_args: &[BasicValueEnum<'c>],
    outputs: TypeRow,
) -> Result<inkwell::values::GlobalValue<'c>> {
    let iwc = ctx.iw_context();
    let builder = ctx.builder();

    let mut type_str = String::new();
    for &arg in fn_args {
        type_str.push(match arg.get_type() {
            BasicTypeEnum::IntType(i) if i.get_bit_width() == 1 => 'b',
            BasicTypeEnum::IntType(i) if i.get_bit_width() == 64 => 'i',
            BasicTypeEnum::FloatType(_) => 'f',
            _ => {
                bail!(
                    "Unsupported argument type for GPU call: {:?}",
                    arg.get_type()
                );
            }
        });
    }
    type_str.push(':');
    type_str.push(match &outputs[..] {
        [] => 'v',
        [i] if ctx.llvm_type(i)? == iwc.i64_type().as_basic_type_enum() => 'i',
        [f] if ctx.llvm_type(f)? == iwc.f64_type().as_basic_type_enum() => 'f',
        _ => bail!("GPU calls expect either no outputs, or a single output of type i64 or f64"),
    });
    let type_str_const = builder.build_global_string_ptr(&type_str, "arg_types")?;
    Ok(type_str_const)
}

/// A helper to pack the arguments into a binary blob
/// suitable for passing to the GPU call.
///
/// Returns a pointer to the blob and its size in bytes.
fn pack_arguments<'c, H: HugrView<Node = Node>>(
    ctx: &EmitFuncContext<'c, '_, H>,
    fn_args: &[BasicValueEnum<'c>],
) -> Result<(inkwell::values::PointerValue<'c>, u32)> {
    // append each arg in binary form to the blob
    let iwc = ctx.iw_context();
    let builder = ctx.builder();

    let mut blob_size = 0;
    for &arg in fn_args {
        blob_size += match arg.get_type() {
            BasicTypeEnum::IntType(i) if i.get_bit_width() == 1 => 1,
            BasicTypeEnum::IntType(i) if i.get_bit_width() == 64 => 8,
            BasicTypeEnum::FloatType(_) => 8,
            _ => {
                bail!(
                    "Unsupported argument type for GPU call: {:?}",
                    arg.get_type()
                );
            }
        };
    }

    let blob_ty = iwc.i8_type().array_type(blob_size);
    let blob = builder.build_alloca(blob_ty, "gpu_input_blob")?;
    let mut offset = 0;
    for &arg in fn_args {
        let dest_ptr = unsafe {
            builder.build_in_bounds_gep(
                blob,
                &[
                    iwc.i64_type().const_zero(),
                    iwc.i64_type().const_int(offset as u64, false),
                ],
                "dest_ptr",
            )?
        };
        match arg.get_type() {
            BasicTypeEnum::IntType(i) if i.get_bit_width() == 1 => {
                // extend to i8
                let arg =
                    builder.build_int_z_extend(arg.into_int_value(), iwc.i8_type(), "arg_i8")?;
                builder.build_store(dest_ptr, arg)?;
                offset += 1;
            }
            BasicTypeEnum::IntType(i) if i.get_bit_width() == 64 => {
                let dst_i64_ptr = builder.build_pointer_cast(
                    dest_ptr,
                    iwc.i64_type().ptr_type(AddressSpace::default()),
                    "dest_i64_ptr",
                )?;
                builder.build_store(dst_i64_ptr, arg)?;
                offset += 8;
            }
            BasicTypeEnum::FloatType(_) => {
                let dst_f64_ptr = builder.build_pointer_cast(
                    dest_ptr,
                    iwc.f64_type().ptr_type(AddressSpace::default()),
                    "dest_f64_ptr",
                )?;
                builder.build_store(dst_f64_ptr, arg)?;
                offset += 8;
            }
            // We have already validated types when sizing the blob,
            // so this case is unreachable.
            _ => unreachable!(),
        }
    }
    Ok((blob, blob_size))
}
