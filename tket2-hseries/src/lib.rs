//! Provides a preparation and validation workflow for Hugrs targeting
//! Quantinuum H-series quantum computers.

use std::mem;

use derive_more::{Display, Error, From};
use hugr::{
    algorithms::{
        const_fold::{ConstFoldError, ConstantFoldPass},
        force_order, remove_polyfuncs,
        validation::{ValidatePassError, ValidationLevel},
        MonomorphizeError, MonomorphizePass,
    },
    hugr::HugrError,
    Hugr, HugrView,
};
use tket2::Tk2Op;

use extension::{
    futures::FutureOpDef,
    qsystem::{LowerTk2Error, LowerTket2ToQSystemPass, QSystemOp},
};
use lazify_measure::{LazifyMeasurePass, LazifyMeasurePassError};

#[cfg(feature = "cli")]
pub mod cli;
pub mod extension;

pub mod lazify_measure;

/// Modify a [hugr::Hugr] into a form that is acceptable for ingress into a Q-System.
/// Returns an error if this cannot be done.
///
/// To construct a `QSystemPass` use [Default::default].
#[derive(Debug, Clone, Copy)]
pub struct QSystemPass {
    validation_level: ValidationLevel,
    constant_fold: bool,
    monomorphize: bool,
}

impl Default for QSystemPass {
    fn default() -> Self {
        Self {
            validation_level: ValidationLevel::default(),
            constant_fold: false,
            monomorphize: true,
        }
    }
}

#[derive(Error, Debug, Display, From)]
#[non_exhaustive]
/// An error reported from [QSystemPass].
pub enum QSystemPassError {
    /// The [hugr::Hugr] was invalid either before or after a pass ran.
    ValidationError(ValidatePassError),
    /// An error from the component [LazifyMeasurePass].
    LazyMeasureError(LazifyMeasurePassError),
    /// An error from the component [force_order()] pass.
    ForceOrderError(HugrError),
    /// An error from the component [LowerTket2ToQSystemPass] pass.
    LowerTk2Error(LowerTk2Error),
    /// An error from the component [ConstantFoldPass] pass.
    ConstantFoldError(ConstFoldError),
    /// An error from the component [MonomorphizePass] pass.
    MonomorphizeError(MonomorphizeError),
}

impl QSystemPass {
    /// Run `QSystemPass` on the given [Hugr]. `registry` is used for
    /// validation, if enabled.
    pub fn run(&self, hugr: &mut Hugr) -> Result<(), QSystemPassError> {
        if self.monomorphize {
            self.monomorphization().run(hugr)?;
            self.validation_level.run_validated_pass(hugr, |hugr, _| {
                let mut owned_hugr = Hugr::default();
                mem::swap(&mut owned_hugr, hugr);
                owned_hugr = remove_polyfuncs(owned_hugr);
                mem::swap(&mut owned_hugr, hugr);
                Ok::<_, QSystemPassError>(())
            })?;
        }

        if self.constant_fold {
            self.constant_fold().run(hugr)?;
        }
        self.lower_tk2().run(hugr)?;
        self.lazify_measure().run(hugr)?;
        self.validation_level.run_validated_pass(hugr, |hugr, _| {
            force_order(hugr, hugr.root(), |hugr, node| {
                let optype = hugr.get_optype(node);
                if optype.cast::<Tk2Op>().is_some() || optype.cast::<QSystemOp>().is_some() {
                    // quantum ops are lifted as early as possible
                    -1
                } else if let Some(FutureOpDef::Read) = hugr.get_optype(node).cast() {
                    // read ops are sunk as late as possible
                    1
                } else {
                    0
                }
            })?;
            Ok::<_, QSystemPassError>(())
        })?;
        Ok(())
    }

    fn lower_tk2(&self) -> LowerTket2ToQSystemPass {
        LowerTket2ToQSystemPass::default().with_validation_level(self.validation_level)
    }

    fn lazify_measure(&self) -> LazifyMeasurePass {
        LazifyMeasurePass::default().with_validation_level(self.validation_level)
    }

    fn constant_fold(&self) -> ConstantFoldPass {
        ConstantFoldPass::default().validation_level(self.validation_level)
    }

    fn monomorphization(&self) -> MonomorphizePass {
        MonomorphizePass::default().validation_level(self.validation_level)
    }

    /// Returns a new `QSystemPass` with the given [ValidationLevel].
    pub fn with_validation_level(mut self, level: ValidationLevel) -> Self {
        self.validation_level = level;
        self
    }

    /// Returns a new `QSystemPass` with constant folding enabled according to
    /// `constant_fold`.
    pub fn with_constant_fold(mut self, constant_fold: bool) -> Self {
        self.constant_fold = constant_fold;
        self
    }

    /// Returns a new `QSystemPass` with monomorphization enabled according to
    /// `monomorphize`.
    pub fn with_monormophize(mut self, monomorphize: bool) -> Self {
        self.monomorphize = monomorphize;
        self
    }
}

#[cfg(test)]
mod test {
    use hugr::{
        builder::{Container, DFGBuilder, Dataflow, DataflowHugr, DataflowSubContainer},
        extension::prelude::{bool_t, qb_t},
        ops::handle::NodeHandle,
        std_extensions::arithmetic::float_types::ConstF64,
        type_row,
        types::Signature,
        HugrView as _,
    };
    use itertools::Itertools as _;
    use petgraph::visit::{Topo, Walker as _};

    use crate::{
        extension::{futures::FutureOpDef, qsystem::QSystemOp},
        QSystemPass,
    };

    #[test]
    fn qsystem_pass() {
        let (mut hugr, [call_node, h_node, f_node, rx_node]) = {
            let mut builder =
                DFGBuilder::new(Signature::new(qb_t(), vec![bool_t(), bool_t()])).unwrap();
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
