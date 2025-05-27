//! LLVM codegen for the `tket2.qsystem.utils` extension.
use tket2::hugr::{self};

use crate::extension::utils::UtilsOp;
use anyhow::Result;
use hugr::llvm::custom::CodegenExtension;
use hugr::llvm::emit::func::EmitFuncContext;
use hugr::llvm::emit::EmitOpArgs;
use tket2::hugr::ops::ExtensionOp;
use tket2::hugr::{HugrView, Node};

/// Codegen extension for `tket2_hseries` `utils` extension.
pub struct UtilsCodegenExtension;

impl CodegenExtension for UtilsCodegenExtension {
    fn add_extension<'a, H: HugrView<Node = Node> + 'a>(
        self,
        builder: tket2::hugr::llvm::CodegenExtsBuilder<'a, H>,
    ) -> tket2::hugr::llvm::CodegenExtsBuilder<'a, H>
    where
        Self: 'a,
    {
        builder.simple_extension_op(|context, args, op| emit_utils_op(context, args, op))
    }
}

/// Lower the `tket2_hseries` `utils` extension.
fn emit_utils_op<H: HugrView<Node = Node>>(
    ctx: &EmitFuncContext<'_, '_, H>,
    args: EmitOpArgs<'_, '_, ExtensionOp, H>,
    op: UtilsOp,
) -> Result<()> {
    match op {
        UtilsOp::GetCurrentShot => {
            let fn_get_cur_shot = ctx.get_extern_func(
                // fun get_current_shot() -> uint
                "get_current_shot",
                ctx.typing_session()
                    .iw_context()
                    .i64_type()
                    .fn_type(&[], false),
            )?;
            let result = ctx
                .builder()
                .build_call(fn_get_cur_shot, &[], "shot")?
                .try_as_basic_value()
                .unwrap_left();
            args.outputs.finish(ctx.builder(), [result])
        }
    }
}
