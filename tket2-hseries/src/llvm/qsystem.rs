//! LLVM lowering implementations for "tket2.qystem" operations.

use tket2::hugr::{self, llvm::inkwell};

use crate::extension::qsystem::{self, QSystemOp};
use anyhow::{anyhow, Result};
use hugr::extension::prelude::qb_t;
use hugr::llvm::custom::CodegenExtension;
use hugr::llvm::emit::func::{build_option, EmitFuncContext};
use hugr::llvm::emit::{emit_value, EmitOpArgs};
use inkwell::builder::Builder;
use inkwell::context::Context;
use inkwell::types::{BasicType, FloatType, IntType};
use inkwell::values::{BasicValueEnum, FunctionValue, IntValue};
use tket2::hugr::llvm::CodegenExtsBuilder;
use tket2::hugr::ops::constant::Value;
use tket2::hugr::ops::ExtensionOp;
use tket2::hugr::{HugrView, Node};

use super::futures::future_type;

/// Codegen extension for quantum operations.
pub struct QSystemCodegenExtension;

impl CodegenExtension for QSystemCodegenExtension {
    fn add_extension<'a, H: HugrView<Node = Node> + 'a>(
        self,
        builder: CodegenExtsBuilder<'a, H>,
    ) -> CodegenExtsBuilder<'a, H>
    where
        Self: 'a,
    {
        builder
            .simple_extension_op(|context, args, op| QSystemEmitter(context).emit(args, op))
            .extension_op(qsystem::EXTENSION_ID, qsystem::RUNTIME_BARRIER_NAME, {
                move |context, args| {
                    // Do nothing
                    // TODO don't lower to RuntimeBarrier
                    args.outputs.finish(context.builder(), args.inputs)
                }
            })
    }
}

struct QSystemEmitter<'c, 'd, 'e, H: HugrView>(&'d mut EmitFuncContext<'c, 'e, H>);

impl<'c, H: HugrView<Node = Node>> QSystemEmitter<'c, '_, '_, H> {
    fn iw_context(&self) -> &'c Context {
        self.0.typing_session().iw_context()
    }

    fn int_type(&self) -> IntType<'c> {
        self.iw_context().i64_type()
    }

    fn qb_type(&self) -> IntType<'c> {
        self.int_type()
    }

    fn float_type(&self) -> FloatType<'c> {
        self.iw_context().f64_type()
    }

    fn bool_type(&self) -> IntType<'c> {
        self.iw_context().bool_type()
    }

    fn builder(&self) -> &Builder<'c> {
        self.0.builder()
    }

    fn get_func_rz(&self) -> Result<FunctionValue<'c>> {
        let func_type = self
            .iw_context()
            .void_type()
            .fn_type(&[self.qb_type().into(), self.float_type().into()], false);
        self.0.get_extern_func("___rz", func_type)
    }

    fn get_func_rzz(&self) -> Result<FunctionValue<'c>> {
        let func_type = self.iw_context().void_type().fn_type(
            &[
                self.qb_type().into(),
                self.qb_type().into(),
                self.float_type().into(),
            ],
            false,
        );
        self.0.get_extern_func("___rzz", func_type)
    }

    fn get_func_rxy(&self) -> Result<FunctionValue<'c>> {
        let func_type = self.iw_context().void_type().fn_type(
            &[
                self.qb_type().into(),
                self.float_type().into(),
                self.float_type().into(),
            ],
            false,
        );
        self.0.get_extern_func("___rxy", func_type)
    }

    fn get_func_qalloc(&self) -> Result<FunctionValue<'c>> {
        let func_type = self.qb_type().fn_type(&[], false);
        self.0.get_extern_func("___qalloc", func_type)
    }

    fn get_func_qfree(&self) -> Result<FunctionValue<'c>> {
        let func_type = self
            .iw_context()
            .void_type()
            .fn_type(&[self.qb_type().into()], false);
        self.0.get_extern_func("___qfree", func_type)
    }

    fn get_func_measure(&self) -> Result<FunctionValue<'c>> {
        let func_type = self.bool_type().fn_type(&[self.qb_type().into()], false);
        self.0.get_extern_func("___measure", func_type)
    }

    fn get_func_lazy_measure(&self) -> Result<FunctionValue<'c>> {
        let func_type = future_type(self.iw_context()).fn_type(&[self.qb_type().into()], false);
        self.0.get_extern_func("___lazy_measure", func_type)
    }

    fn get_func_reset(&self) -> Result<FunctionValue<'c>> {
        let func_type = self
            .iw_context()
            .void_type()
            .fn_type(&[self.qb_type().into()], false);
        self.0.get_extern_func("___reset", func_type)
    }

    /// Helper function to `emit` a qsystem operation.
    fn emit_op(
        &self,
        args: EmitOpArgs<'c, '_, ExtensionOp, H>,
        name: &str,
        func: Result<FunctionValue<'c>>,
        input_indices: &[usize],
        output_indices: &[usize],
    ) -> Result<()> {
        let inputs: Vec<_> = input_indices.iter().map(|&i| args.inputs[i]).collect();
        let outputs: Vec<_> = output_indices.iter().map(|&i| args.inputs[i]).collect();
        self.builder().build_call(
            func?,
            &inputs.iter().map(|&v| v.into()).collect::<Vec<_>>(),
            name,
        )?;
        args.outputs.finish(self.builder(), outputs)
    }

    /// Function to help lower the `tket2.qsystem` extension.
    fn emit(&mut self, args: EmitOpArgs<'c, '_, ExtensionOp, H>, op: QSystemOp) -> Result<()> {
        match op {
            // Rotation about Z
            QSystemOp::Rz => self.emit_op(args, "rz", self.get_func_rz(), &[0, 1], &[0]),
            // Rotation about ZZ (aka "ZZPhase")
            QSystemOp::ZZPhase => {
                self.emit_op(args, "rzz", self.get_func_rzz(), &[0, 1, 2], &[0, 1])
            }
            // Rotation in X and Y
            // (α,β) ↦ Rz(β)Rx(α)Rz(−β)
            //
            // aka "PhasedX", "R1XY", "U1q"
            QSystemOp::PhasedX => self.emit_op(args, "rxy", self.get_func_rxy(), &[0, 1, 2], &[0]),
            // Measure qubit in Z basis
            QSystemOp::Measure | QSystemOp::MeasureReset => {
                let [qb] = args
                    .inputs
                    .try_into()
                    .map_err(|_| anyhow!("Measure expects one input"))?;
                let measure = self.get_func_measure()?;
                let result_i1 = self
                    .builder()
                    .build_call(measure, &[qb.into()], "measure_i1")?
                    .try_as_basic_value()
                    .unwrap_left()
                    .into_int_value();
                let true_val = emit_value(self.0, &Value::true_val())?;
                let false_val = emit_value(self.0, &Value::false_val())?;
                let result = self
                    .builder()
                    .build_select(result_i1, true_val, false_val, "measure")?;
                if op == QSystemOp::Measure {
                    // normal measure may put the qubit in invalid state, so assume
                    // deallocation, don't return it
                    self.builder()
                        .build_call(self.get_func_qfree()?, &[qb.into()], "qfree")?;
                    args.outputs.finish(self.builder(), [result])
                } else {
                    // MeasureReset will reset the qubit after measurement so safe to return
                    self.builder()
                        .build_call(self.get_func_reset()?, &[qb.into()], "reset")?;
                    args.outputs.finish(self.builder(), [qb, result])
                }
            }
            // Measure qubit in Z basis, not forcing to a boolean
            QSystemOp::LazyMeasure => {
                let [qb] = args
                    .inputs
                    .try_into()
                    .map_err(|_| anyhow!("LazyMeasure expects one input"))?;
                let result = self
                    .builder()
                    .build_call(self.get_func_lazy_measure()?, &[qb.into()], "lazy_measure")?
                    .try_as_basic_value()
                    .unwrap_left();
                let qfree = self.get_func_qfree()?;
                self.builder().build_call(qfree, &[qb.into()], "qfree")?;
                args.outputs.finish(self.builder(), [result])
            }
            QSystemOp::LazyMeasureReset => {
                let [qb] = args
                    .inputs
                    .try_into()
                    .map_err(|_| anyhow!("LazyMeasureReset expects one input"))?;
                let measure = self.get_func_lazy_measure()?;
                let result = self
                    .builder()
                    .build_call(measure, &[qb.into()], "lazy_measure")?
                    .try_as_basic_value()
                    .unwrap_left();
                let reset = self.get_func_reset()?;
                self.builder().build_call(reset, &[qb.into()], "reset")?;
                args.outputs.finish(self.builder(), [qb, result])
            }
            // Reset a qubit
            QSystemOp::Reset => self.emit_op(args, "reset", self.get_func_reset(), &[0], &[0]),
            QSystemOp::TryQAlloc => {
                let [] = args
                    .inputs
                    .try_into()
                    .map_err(|_| anyhow!("QAlloc expects no inputs"))?;
                let qb = self
                    .builder()
                    .build_call(self.get_func_qalloc()?, &[], "qalloc")?
                    .try_as_basic_value()
                    .unwrap_left();

                let max_qb = self.qb_type().const_int(u64::MAX, false);
                let not_max = self.builder().build_int_compare(
                    inkwell::IntPredicate::NE,
                    qb.into_int_value(),
                    max_qb,
                    "not_max",
                )?;
                self.reset_if_qb(qb, not_max)?;

                let result = build_option(self.0, not_max, qb, qb_t())?;
                args.outputs.finish(self.builder(), [result])
            }
            QSystemOp::QFree => self.emit_op(args, "qfree", self.get_func_qfree(), &[0], &[]),
        }
    }

    /// Reset a qubit if it is successfully allocated (not max value)
    fn reset_if_qb(
        &self,
        qb: BasicValueEnum<'c>,
        not_max: IntValue<'_>,
    ) -> Result<(), anyhow::Error> {
        let orig_bb = self
            .builder()
            .get_insert_block()
            .ok_or_else(|| anyhow!("No current insertion point"))?;

        let reset_bb = self
            .iw_context()
            .insert_basic_block_after(orig_bb, "reset_bb");
        let id_bb = self
            .iw_context()
            .insert_basic_block_after(reset_bb, "id_bb");
        let reset_bb = {
            self.builder().position_at_end(reset_bb);
            let reset = self.get_func_reset()?;
            self.builder().build_call(reset, &[qb.into()], "reset")?;
            self.builder().build_unconditional_branch(id_bb)?;
            reset_bb
        };

        self.builder().position_at_end(orig_bb);
        self.builder()
            .build_conditional_branch(not_max, reset_bb, id_bb)?;
        self.builder().position_at_end(id_bb);
        Ok(())
    }
}
