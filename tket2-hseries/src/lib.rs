//! Provides a preparation and validation workflow for Hugrs targeting
//! Quantinuum H-series quantum computers.

use derive_more::{Display, Error, From};
use hugr::{
    algorithms::{
        const_fold::{ConstFoldError, ConstantFoldPass},
        force_order,
        replace_types::ReplaceTypesError,
        ComposablePass as _, LinearizeArrayPass, MonomorphizePass, RemoveDeadFuncsError,
        RemoveDeadFuncsPass,
    },
    hugr::HugrError,
    Hugr, HugrView, Node,
};
use replace_bools::{ReplaceBoolPass, ReplaceBoolPassError};
use tket2::Tk2Op;

use extension::{
    futures::FutureOpDef,
    qsystem::{LowerTk2Error, LowerTket2ToQSystemPass, QSystemOp},
};

#[cfg(feature = "cli")]
pub mod cli;
pub mod extension;

pub mod replace_bools;

/// Modify a [hugr::Hugr] into a form that is acceptable for ingress into a Q-System.
/// Returns an error if this cannot be done.
///
/// To construct a `QSystemPass` use [Default::default].
#[derive(Debug, Clone, Copy)]
pub struct QSystemPass {
    constant_fold: bool,
    monomorphize: bool,
    force_order: bool,
    lazify: bool,
}

impl Default for QSystemPass {
    fn default() -> Self {
        Self {
            constant_fold: false,
            monomorphize: true,
            force_order: true,
            lazify: true,
        }
    }
}

#[derive(Error, Debug, Display, From)]
#[non_exhaustive]
/// An error reported from [QSystemPass].
pub enum QSystemPassError<N = Node> {
    /// An error from the component [ReplaceBoolPass].
    ReplaceBoolError(ReplaceBoolPassError<N>),
    /// An error from the component [force_order()] pass.
    ForceOrderError(HugrError),
    /// An error from the component [LowerTket2ToQSystemPass] pass.
    LowerTk2Error(LowerTk2Error),
    /// An error from the component [ConstantFoldPass] pass.
    ConstantFoldError(ConstFoldError),
    /// An error from the component [LinearizeArrayPass] pass.
    LinearizeArrayError(ReplaceTypesError),
    /// An error when running [RemoveDeadFuncsPass] after the monomorphisation
    /// pass.
    ///
    ///  [RemoveDeadFuncsPass]: hugr::algorithms::RemoveDeadFuncsError
    DCEError(RemoveDeadFuncsError),
    /// No [FuncDefn] named "main" in [Module].
    ///
    /// [FuncDefn]: hugr::ops::FuncDefn
    /// [Module]: hugr::ops::Module
    #[display("No function named 'main' in module.")]
    NoMain,
}

impl QSystemPass {
    /// Run `QSystemPass` on the given [Hugr]. `registry` is used for
    /// validation, if enabled.
    pub fn run(&self, hugr: &mut Hugr) -> Result<(), QSystemPassError> {
        if self.monomorphize {
            self.monomorphization().run(hugr).unwrap();

            let mut rdfp = RemoveDeadFuncsPass::default();
            if hugr.entrypoint_optype().is_module() {
                let main_node = hugr
                    .children(hugr.entrypoint())
                    .find(|&n| {
                        hugr.get_optype(n)
                            .as_func_defn()
                            .is_some_and(|fd| fd.func_name() == "main")
                    })
                    .ok_or(QSystemPassError::NoMain)?;
                rdfp = rdfp.with_module_entry_points([main_node]);
            }
            rdfp.run(hugr)?
        }

        self.lower_tk2().run(hugr)?;
        if self.lazify {
            self.replace_bools().run(hugr)?;
        }
        self.linearize_arrays().run(hugr)?;
        if self.constant_fold {
            self.constant_fold().run(hugr)?;
        }
        if self.force_order {
            self.force_order(hugr)?;
        }
        Ok(())
    }

    fn force_order(&self, hugr: &mut Hugr) -> Result<(), QSystemPassError> {
        force_order(hugr, hugr.entrypoint(), |hugr, node| {
            let optype = hugr.get_optype(node);

            let is_quantum =
                optype.cast::<Tk2Op>().is_some() || optype.cast::<QSystemOp>().is_some();
            let is_qalloc = matches!(optype.cast(), Some(Tk2Op::QAlloc) | Some(Tk2Op::TryQAlloc))
                || optype.cast() == Some(QSystemOp::TryQAlloc);
            let is_qfree =
                optype.cast() == Some(Tk2Op::QFree) || optype.cast() == Some(QSystemOp::QFree);
            let is_read = optype.cast() == Some(FutureOpDef::Read);

            // HACK: for now qallocs and qfrees are not adequately ordered,
            // see <https://github.com/CQCL/guppylang/issues/778>. To
            // mitigate this we push qfrees as early as possible and qallocs
            // as late as possible
            //
            // To maximise laziness we push quantum ops (including
            // LazyMeasure) as early as possible and Future::Read as late as
            // possible.
            if is_qfree {
                -3
            } else if is_quantum && !is_qalloc {
                // non-qalloc quantum ops
                -2
            } else if is_qalloc {
                -1
            } else if !is_read {
                // all other ops
                0
            } else {
                // Future::Read ops
                1
            }
        })?;
        Ok::<_, QSystemPassError>(())
    }

    fn lower_tk2(&self) -> LowerTket2ToQSystemPass {
        LowerTket2ToQSystemPass
    }

    fn replace_bools(&self) -> ReplaceBoolPass {
        ReplaceBoolPass
    }

    fn constant_fold(&self) -> ConstantFoldPass {
        ConstantFoldPass::default()
    }

    fn monomorphization(&self) -> MonomorphizePass {
        MonomorphizePass
    }

    fn linearize_arrays(&self) -> LinearizeArrayPass {
        LinearizeArrayPass::default()
    }

    /// Returns a new `QSystemPass` with constant folding enabled according to
    /// `constant_fold`.
    ///
    /// Off by default.
    pub fn with_constant_fold(mut self, constant_fold: bool) -> Self {
        self.constant_fold = constant_fold;
        self
    }

    /// Returns a new `QSystemPass` with monomorphization enabled according to
    /// `monomorphize`.
    ///
    /// On by default.
    pub fn with_monormophize(mut self, monomorphize: bool) -> Self {
        self.monomorphize = monomorphize;
        self
    }

    /// Returns a new `QSystemPass` with forcing the HUGR to have
    /// totally-ordered ops enabled according to `force_order`.
    ///
    /// On by default.
    ///
    /// When enabled, we push quantum ops as early as possible, and we push
    /// `tket2.futures.read` ops as late as possible.
    pub fn with_force_order(mut self, force_order: bool) -> Self {
        self.force_order = force_order;
        self
    }

    /// Returns a new `QSystemPass` with lazification enabled according to `lazify`.
    ///
    /// On by default.
    ///
    /// When enabled we replace strict measurement ops with lazy equivalents
    /// from `tket2.qsystem`.
    pub fn with_lazify(mut self, lazify: bool) -> Self {
        self.lazify = lazify;
        self
    }
}

#[cfg(test)]
mod test {
    use hugr::{
        builder::{Container, DFGBuilder, Dataflow, DataflowHugr, DataflowSubContainer},
        extension::prelude::qb_t,
        ops::handle::NodeHandle,
        std_extensions::arithmetic::float_types::ConstF64,
        type_row,
        types::Signature,
        HugrView as _,
    };

    use itertools::Itertools as _;
    use petgraph::visit::{Topo, Walker as _};
    use tket2::extension::bool::bool_type;

    use crate::{
        extension::{futures::FutureOpDef, qsystem::QSystemOp},
        QSystemPass,
    };

    #[test]
    fn qsystem_pass() {
        let (mut hugr, [call_node, h_node, f_node, rx_node]) = {
            let mut builder =
                DFGBuilder::new(Signature::new(qb_t(), vec![bool_type(), bool_type()])).unwrap();
            let func = builder
                .define_function("func", Signature::new_endo(type_row![]))
                .unwrap()
                .finish_with_outputs([])
                .unwrap();
            let [qb] = builder.input_wires_arr();

            // This call node has no dependencies, so it should be lifted above
            // Future Reads and sunk below quantum ops.
            let call_node = builder.call(func.handle(), &[], []).unwrap().node();

            // this LoadConstant should be pushed below the quantum ops where possible
            let angle = builder.add_load_value(ConstF64::new(0.0));
            let f_node = angle.node();

            // with no dependencies, this Reset should be lifted to the start
            let [qb] = builder
                .add_dataflow_op(QSystemOp::Reset, [qb])
                .unwrap()
                .outputs_arr();
            let h_node = qb.node();

            // depending on the angle means this op can't be lifted above the angle ops
            let [qb] = builder
                .add_dataflow_op(QSystemOp::Rz, [qb, angle])
                .unwrap()
                .outputs_arr();
            let rx_node = qb.node();

            // the Measure node will be removed. A Lazy Measure and two Future
            // Reads will be added.  The Lazy Measure will be lifted and the
            // reads will be sunk.
            let [measure_result] = builder
                .add_dataflow_op(QSystemOp::Measure, [qb])
                .unwrap()
                .outputs_arr();

            let hugr = builder
                .finish_hugr_with_outputs([measure_result, measure_result])
                .unwrap();
            (hugr, [call_node, h_node, f_node, rx_node])
        };

        QSystemPass::default().run(&mut hugr).unwrap();

        let topo_sorted = Topo::new(&hugr.as_petgraph())
            .iter(&hugr.as_petgraph())
            .collect_vec();

        let get_pos = |x| topo_sorted.iter().position(|&y| y == x).unwrap();
        assert!(get_pos(h_node) < get_pos(f_node));
        assert!(get_pos(h_node) < get_pos(call_node));
        assert!(get_pos(rx_node) < get_pos(call_node));

        for &n in topo_sorted
            .iter()
            .filter(|&&n| FutureOpDef::try_from(hugr.get_optype(n)) == Ok(FutureOpDef::Read))
        {
            assert!(get_pos(call_node) < get_pos(n));
        }
    }
}
