//! LLVM lowering implementations for HUGR prelude operations.

use std::hash::{DefaultHasher, Hash as _, Hasher as _};
use tket2::hugr::{self, llvm::inkwell};

use anyhow::{anyhow, ensure, Result};
use hugr::extension::prelude::ConstString;
use hugr::llvm::emit::emit_value;
use hugr::llvm::emit::func::EmitFuncContext;
use hugr::llvm::extension::prelude::PreludeCodegen;
use hugr::llvm::types::TypingSession;
use inkwell::attributes::{Attribute, AttributeLoc};
use inkwell::module::Linkage;
use inkwell::types::{BasicType, IntType};
use inkwell::values::{BasicValue, BasicValueEnum, FunctionValue, StructValue};
use inkwell::AddressSpace;
use tket2::hugr::{HugrView, Node};

#[derive(Clone)]
/// Codegen extension for the QIS prelude.
pub struct QISPreludeCodegen;

impl PreludeCodegen for QISPreludeCodegen {
    fn usize_type<'c>(&self, session: &TypingSession<'c, '_>) -> IntType<'c> {
        session.iw_context().i64_type()
    }

    /// Return the llvm type of [`tket2::hugr::extension::prelude::qb_t`].
    fn qubit_type<'c>(&self, session: &TypingSession<'c, '_>) -> impl BasicType<'c> {
        session.iw_context().i64_type()
    }

    /// A panic ends all shots, this is done by using a return code > 1000.
    /// The lowering of the panic HUGR op shifts the return code held by the op by 1000.
    /// So valid values for `signal` in the panic HUGR op are 1-(2^31 - 1001).
    fn emit_panic<H: HugrView<Node = Node>>(
        &self,
        ctx: &mut EmitFuncContext<H>,
        err: BasicValueEnum,
    ) -> Result<()> {
        let err = StructValue::try_from(err)
            .map_err(|()| anyhow!("emit_panic: Expected err to have struct type"))?;
        let return_code = ctx.builder().build_extract_value(err, 0, "")?;
        // TODO check if code is in valid range 1-(2^31 - 1001), otherwise error.
        // https://github.com/quantinuum-dev/eldarion/issues/287
        let return_code = {
            let shift = ctx.iw_context().i32_type().const_int(1000, false);
            ctx.builder()
                .build_int_add(return_code.into_int_value(), shift, "shift_code")?
        };

        let msg = ctx.builder().build_extract_value(err, 1, "")?;
        let panic = Self::get_panic(ctx)?;
        ctx.builder()
            .build_call(panic, &[return_code.into(), msg.into()], "")?;
        Ok(())
    }

    /// An exit ends the current shot using a return code `rc`, where `1 <= rc <= 1000`.
    fn emit_exit<H: HugrView<Node = Node>>(
        &self,
        ctx: &mut EmitFuncContext<H>,
        err: BasicValueEnum,
    ) -> Result<()> {
        let err = StructValue::try_from(err)
            .map_err(|()| anyhow!("emit_exit: Expected err to have struct type"))?;
        let return_code = ctx.builder().build_extract_value(err, 0, "")?;
        // TODO check if code is in valid range 1-1000, otherwise error.
        // https://github.com/quantinuum-dev/eldarion/issues/287
        let msg = ctx.builder().build_extract_value(err, 1, "")?;
        let panic = Self::get_panic(ctx)?;
        ctx.builder()
            .build_call(panic, &[return_code.into(), msg.into()], "")?;
        Ok(())
    }

    fn emit_print<H: HugrView<Node = Node>>(
        &self,
        _ctx: &mut EmitFuncContext<H>,
        _text: inkwell::values::BasicValueEnum,
    ) -> Result<()> {
        Ok(())
    }

    fn emit_const_error<'c, H: HugrView<Node = Node>>(
        &self,
        ctx: &mut EmitFuncContext<'c, '_, H>,
        err: &hugr::extension::prelude::ConstError,
    ) -> Result<BasicValueEnum<'c>> {
        let err_ty = ctx
            .llvm_type(&hugr::extension::prelude::error_type())?
            .into_struct_type();
        let signal = err_ty
            .get_field_type_at_index(0)
            .unwrap()
            .into_int_type()
            .const_int(u64::from(err.signal), false);

        let message = emit_value(ctx, &ConstString::new(err.message.clone()).into())?;
        let err = err_ty.const_named_struct(&[signal.into(), message]);
        Ok(err.into())
    }

    fn emit_const_string<'c, H: HugrView<Node = Node>>(
        &self,
        ctx: &mut EmitFuncContext<'c, '_, H>,
        str: &hugr::extension::prelude::ConstString,
    ) -> Result<BasicValueEnum<'c>> {
        emit_global_string(ctx, str.value(), "s_", "INT")
    }
}

impl QISPreludeCodegen {
    fn get_panic<'c, H: HugrView<Node = Node>>(
        ctx: &EmitFuncContext<'c, '_, H>,
    ) -> Result<FunctionValue<'c>> {
        let iwc = ctx.iw_context();
        let func_type = iwc.void_type().fn_type(
            &[
                iwc.i32_type().into(),
                iwc.i8_type().ptr_type(AddressSpace::default()).into(),
            ],
            false,
        );
        let f = ctx.get_extern_func("panic", func_type)?;
        let noreturn = Attribute::get_named_enum_kind_id("noreturn");
        debug_assert!(noreturn != 0);
        f.add_attribute(
            AttributeLoc::Function,
            iwc.create_enum_attribute(noreturn, 0),
        );

        Ok(f)
    }
}
// TODO check
static TAG_PREFIX: &str = "USER";

/// Emit a global string constant with a unique name.
/// The string is prefixed with a tag and the type tag.
pub fn emit_global_string<'c, H: HugrView<Node = Node>>(
    ctx: &EmitFuncContext<'c, '_, H>,
    str: impl AsRef<str>,
    symbol_prefix: impl AsRef<str>,
    type_tag: impl AsRef<str>,
) -> Result<BasicValueEnum<'c>> {
    let (str, symbol_prefix, type_tag) = (str.as_ref(), symbol_prefix.as_ref(), type_tag.as_ref());
    let tagged_str = format!("{TAG_PREFIX}:{type_tag}:{str}");
    let tagged_str_bytes = tagged_str.as_bytes();
    let tagged_str_len = tagged_str_bytes.len();
    ensure!(
        tagged_str_len < 256,
        "Constant string too long: {tagged_str_len} >= 256"
    );

    let tgt_str_bytes = [&[u8::try_from(tagged_str_len)?], tagged_str_bytes].concat();
    let tgt_str_value = ctx.iw_context().const_string(&tgt_str_bytes, false);

    let global = {
        // we need a unique name for the global. We use a prefix(of length
        // 10) of the string, followed by the hash of the whole string.
        //
        // This should be enough, but for absolute guarantee of success we
        // append ".i", i an integer and find an available name.
        let hash = {
            let mut s = DefaultHasher::default();
            tgt_str_bytes.hash(&mut s);
            let h = s.finish();
            let lower32 = h as u32;
            format!("{lower32:08X}")
        };
        let str_len = str.len();
        let name_prefix = format!("{symbol_prefix}{}.{hash}", &str[..10.min(str_len)]);

        (0..)
            .find_map(|i| {
                let name = format!("{name_prefix}.{i}");
                if let Some(global) = ctx.get_current_module().get_global(&name) {
                    if let Some(v) = global.get_initializer() {
                        if v == tgt_str_value {
                            // We've found a global with the same value, use that.
                            return Some(global);
                        }
                    }
                } else {
                    let global =
                        ctx.get_current_module()
                            .add_global(tgt_str_value.get_type(), None, &name);
                    global.set_initializer(&tgt_str_value);
                    global.set_linkage(Linkage::Private);
                    global.set_constant(true);
                    return Some(global);
                }
                None
            })
            .unwrap()
    };
    Ok(ctx
        .builder()
        .build_pointer_cast(
            global.as_pointer_value(),
            ctx.iw_context().i8_type().ptr_type(AddressSpace::default()),
            "",
        )?
        .as_basic_value_enum())
}
