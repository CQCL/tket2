//! LLVM lowering implementations for the "tket2.qsystem.random" extension.

use tket2::hugr::{self, llvm::inkwell};

use crate::extension::random::{self, RandomOp, CONTEXT_TYPE_NAME};
use anyhow::{anyhow, Result};
use hugr::llvm::custom::CodegenExtension;
use hugr::llvm::emit::func::EmitFuncContext;
use hugr::llvm::emit::EmitOpArgs;
use inkwell::builder::Builder;
use inkwell::context::Context;
use inkwell::types::{BasicTypeEnum, FloatType, IntType};
use inkwell::values::{BasicValueEnum, FunctionValue};
use tket2::hugr::ops::ExtensionOp;
use tket2::hugr::{HugrView, Node};

/// Codegen extension for `tket2.qsystem.random` extension.
pub struct RandomCodegenExtension;

impl CodegenExtension for RandomCodegenExtension {
    fn add_extension<'a, H: HugrView<Node = Node> + 'a>(
        self,
        builder: tket2::hugr::llvm::CodegenExtsBuilder<'a, H>,
    ) -> tket2::hugr::llvm::CodegenExtsBuilder<'a, H>
    where
        Self: 'a,
    {
        builder
            .custom_type(
                (random::EXTENSION_ID, CONTEXT_TYPE_NAME.to_owned()),
                |session, hugr_type| match (hugr_type.name().as_str(), hugr_type.args()) {
                    ("context", _) => Ok(rng_context_type(session.iw_context())),
                    _ => Err(anyhow!(
                        "RandomCodegenExtension: Unsupported type: {}",
                        hugr_type
                    )),
                },
            )
            .simple_extension_op(|context, args, op| RandomEmitter(context).emit(args, op))
    }
}

fn rng_context_type(context: &Context) -> BasicTypeEnum {
    context.struct_type(&[], false).into()
}

/// Lower the `tket2.qsystem.random` extension.
struct RandomEmitter<'c, 'd, 'e, H: HugrView<Node = Node>>(&'d mut EmitFuncContext<'c, 'e, H>);

impl<'c, H: HugrView<Node = Node>> RandomEmitter<'c, '_, '_, H> {
    fn iw_context(&self) -> &'c Context {
        self.0.typing_session().iw_context()
    }

    fn i32_type(&self) -> IntType<'c> {
        self.iw_context().i32_type()
    }

    fn i64_type(&self) -> IntType<'c> {
        self.iw_context().i64_type()
    }

    fn float_type(&self) -> FloatType<'c> {
        self.iw_context().f64_type()
    }

    fn bool_type(&self) -> IntType<'c> {
        self.iw_context().bool_type()
    }

    fn rng_context(&self) -> BasicValueEnum<'c> {
        self.iw_context().const_struct(&[], false).into()
    }

    fn builder(&self) -> &Builder<'c> {
        self.0.builder()
    }

    /// Helper function to `emit` an RNG operation.
    fn emit_op(
        &self,
        args: EmitOpArgs<'c, '_, ExtensionOp, H>,
        name: &str,
        func: Result<FunctionValue<'c>>,
        input_indices: &[usize],
    ) -> Result<()> {
        let inputs: Vec<_> = input_indices.iter().map(|&i| args.inputs[i]).collect();
        let result = self
            .builder()
            .build_call(
                func?,
                &inputs.iter().map(|&v| v.into()).collect::<Vec<_>>(),
                name,
            )?
            .try_as_basic_value()
            .unwrap_left();
        args.outputs
            .finish(self.builder(), [result, self.rng_context()])
    }

    /// Function to help lower the `tket2.qsystem.random` extension.
    fn emit(&self, args: EmitOpArgs<'c, '_, ExtensionOp, H>, op: RandomOp) -> Result<()> {
        match op {
            RandomOp::RandomInt => {
                let fn_random_int = self
                    .0
                    .get_extern_func("random_int", self.i32_type().fn_type(&[], false));
                self.emit_op(args, "rint", fn_random_int, &[])
            }
            RandomOp::RandomFloat => {
                let fn_random_float = self
                    .0
                    .get_extern_func("random_float", self.float_type().fn_type(&[], false));
                self.emit_op(args, "rfloat", fn_random_float, &[])
            }
            RandomOp::RandomIntBounded => {
                let fn_random_int_bounded = self.0.get_extern_func(
                    "random_rng",
                    self.i32_type().fn_type(&[self.i32_type().into()], false),
                );
                self.emit_op(args, "rintb", fn_random_int_bounded, &[1])
            }
            RandomOp::NewRNGContext => {
                let fn_random_seed = self.0.get_extern_func(
                    "random_seed",
                    self.iw_context()
                        .void_type()
                        .fn_type(&[self.i64_type().into()], false),
                )?;
                let [seed] = args
                    .inputs
                    .try_into()
                    .map_err(|_| anyhow!("NewRNGContext expects a seed argument"))?;
                self.builder()
                    .build_call(fn_random_seed, &[seed.into()], "rseed")?;
                args.outputs.finish(
                    self.builder(),
                    [self.bool_type().const_int(1, false).into()],
                )
            }
            RandomOp::DeleteRNGContext => args.outputs.finish(self.builder(), []),
        }
    }
}
