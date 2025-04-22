use anyhow::{Result, anyhow};
use tket2::hugr::{self, llvm::{self as hugr_llvm, inkwell}};
use hugr_llvm::CodegenExtsBuilder;
use hugr_llvm::custom::CodegenExtension;
use hugr_llvm::emit::func::EmitFuncContext;
use hugr_llvm::emit::{EmitOpArgs, emit_value};
use inkwell::builder::Builder;
use inkwell::context::Context;
use inkwell::types::{BasicTypeEnum, IntType};
use inkwell::values::FunctionValue;
use hugr::extension::prelude::bool_t;
use hugr::extension::simple_op::MakeExtensionOp;
use hugr::ops::{ExtensionOp, Value};
use hugr::types::TypeArg;
use hugr::{HugrView, Node};
use crate::extension::futures::{self, FUTURE_TYPE_NAME, FutureOp, FutureOpDef};

/// Codegen extension for futures
pub struct FuturesCodegenExtension;

impl CodegenExtension for FuturesCodegenExtension {
    fn add_extension<'a, H: HugrView<Node = Node> + 'a>(
        self,
        builder: CodegenExtsBuilder<'a, H>,
    ) -> CodegenExtsBuilder<'a, H>
    where
        Self: 'a,
    {
        builder
            .custom_type(
                (futures::EXTENSION_ID, FUTURE_TYPE_NAME.to_owned()),
                |session, hugr_type| {
                    match (hugr_type.name().as_str(), hugr_type.args()) {
                        // For now, we only support future bools
                        ("Future", [TypeArg::Type { ty }]) if *ty == bool_t() => {
                            Ok(future_type(session.iw_context()))
                        }
                        _ => Err(anyhow!(
                            "FuturesCodegenExtension: Unsupported type: {}",
                            hugr_type
                        )),
                    }
                },
            )
            .simple_extension_op::<FutureOpDef>(|context, args, _op| {
                FuturesEmitter(context).emit(args)
            })
    }
}

pub fn future_type(context: &Context) -> BasicTypeEnum {
    // The runtime represent a future handle as an i64
    context.i64_type().into()
}

struct FuturesEmitter<'c, 'd, 'e, H: HugrView<Node = Node>>(&'d mut EmitFuncContext<'c, 'e, H>);

impl<'c, H: HugrView<Node = Node>> FuturesEmitter<'c, '_, '_, H> {
    fn iw_context(&self) -> &'c Context {
        self.0.typing_session().iw_context()
    }

    fn cl_future_type(&self) -> BasicTypeEnum<'c> {
        future_type(self.iw_context())
    }

    fn cl_bool_type(&self) -> IntType<'c> {
        self.iw_context().bool_type()
    }

    fn builder(&self) -> &Builder<'c> {
        self.0.builder()
    }

    fn get_func_inc_refcount(&self) -> Result<FunctionValue<'c>> {
        let func_type = self
            .iw_context()
            .void_type()
            .fn_type(&[self.cl_future_type().into()], false);
        self.0.get_extern_func("___inc_future_refcount", func_type)
    }

    fn get_func_dec_refcount(&self) -> Result<FunctionValue<'c>> {
        let func_type = self
            .iw_context()
            .void_type()
            .fn_type(&[self.cl_future_type().into()], false);
        self.0.get_extern_func("___dec_future_refcount", func_type)
    }

    fn get_func_read_bool(&self) -> Result<FunctionValue<'c>> {
        let func_type = self
            .cl_bool_type()
            .fn_type(&[self.cl_future_type().into()], false);
        self.0.get_extern_func("___read_future_bool", func_type)
    }

    fn emit(&mut self, args: EmitOpArgs<'c, '_, ExtensionOp, H>) -> Result<()> {
        match FutureOp::from_optype(&args.node().generalise()).unwrap().op {
            FutureOpDef::Dup => {
                let [arg] = args
                    .inputs
                    .try_into()
                    .map_err(|_| anyhow!("Dup expects a single input"))?;
                let func = self.get_func_inc_refcount()?;
                self.builder()
                    .build_call(func, &[arg.into()], "inc_refcount")?;
                args.outputs.finish(self.builder(), [arg, arg])
            }
            FutureOpDef::Free => {
                let [arg] = args
                    .inputs
                    .try_into()
                    .map_err(|_| anyhow!("Free expects a single input"))?;
                let func = self.get_func_dec_refcount()?;
                self.builder()
                    .build_call(func, &[arg.into()], "dec_refcount")?;
                args.outputs.finish(self.builder(), [])
            }
            FutureOpDef::Read => {
                let [arg] = args
                    .inputs
                    .try_into()
                    .map_err(|_| anyhow!("Read expects a single input"))?;
                let read_func = self.get_func_read_bool()?;
                let result_i1 = self
                    .builder()
                    .build_call(read_func, &[arg.into()], "read_bool")?
                    .try_as_basic_value()
                    .unwrap_left()
                    .into_int_value();
                let dec_refcount_func = self.get_func_dec_refcount()?;
                self.builder()
                    .build_call(dec_refcount_func, &[arg.into()], "dec_refcount")?;
                let true_val = emit_value(self.0, &Value::true_val())?;
                let false_val = emit_value(self.0, &Value::false_val())?;
                let result = self
                    .builder()
                    .build_select(result_i1, true_val, false_val, "measure")?;
                args.outputs.finish(self.builder(), [result])
            }
            n => Err(anyhow!("Unknown op {:?}", n)),
        }
    }
}

#[cfg(test)]
mod test {
    use hugr::extension::simple_op::MakeRegisteredOp;
    use hugr_llvm::test::llvm_ctx;
    use hugr_llvm::test::TestContext;
    use hugr_llvm::test::single_op_hugr;
    use hugr_llvm::check_emission;

    use super::*;
    #[rstest::rstest]
    #[case::read(1,FutureOp { op: FutureOpDef::Read, typ: bool_t() })]
    #[case::dup(2,FutureOp { op: FutureOpDef::Dup, typ: bool_t() })]
    #[case::free(3,FutureOp { op: FutureOpDef::Free, typ: bool_t() })]
    fn emit_futures_codegen(
        #[case] _i: i32,
        #[with(_i)] mut llvm_ctx: TestContext,
        #[case] op: FutureOp,
    ) {
        llvm_ctx.add_extensions(|ceb| {
            ceb.add_extension(FuturesCodegenExtension)
        });
        let hugr = single_op_hugr(op.to_extension_op().unwrap().into());
        check_emission!(hugr, llvm_ctx);
    }

}
