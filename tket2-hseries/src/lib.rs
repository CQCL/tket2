//! Provides a preparation and validation workflow for Hugrs targeting
//! Quantinuum H-series quantum computers.
use hugr::{
    algorithms::{
        force_order, validation::{ValidatePassError, ValidationLevel},
    },
    extension::ExtensionRegistry,
    hugr::{hugrmut::HugrMut, HugrError},
};
use tket2::Tk2Op;

use thiserror::Error;

use extension::{futures::FutureOpDef, quantum_lazy::LazyQuantumOp};
use lazify_measure::{LazifyMeasurePass, LazifyMeasurePassError};

#[cfg(feature = "cli")]
pub mod cli;

pub mod extension;

pub mod lazify_measure;

/// Modify a [hugr::Hugr] into a form that is acceptable for ingress into an H-series.
/// Returns an error if this cannot be done.
///
/// To constuct a `HSeriesPass` use [Default::default].
#[derive(Debug, Clone, Copy, Default)]
pub struct HSeriesPass {
    validation_level: ValidationLevel,
}

#[derive(Error, Debug)]
/// An error reported from [HSeriesPass].

pub enum HSeriesPassError {
    /// The [hugr::Hugr] was invalid either before or after a pass ran.
    #[error(transparent)]
    ValidationError(#[from] ValidatePassError),
    /// An error from the component [LazifyMeasurePass].
    #[error(transparent)]
    LazyMeasureError(#[from] LazifyMeasurePassError),
    /// An error from the component [force_order()] pass.
    #[error(transparent)]
    ForceOrderError(#[from] HugrError),
}

impl HSeriesPass {
    /// Run `HSeriesPass` on the given [HugrMut]. `registry` is used for
    /// validation, if enabled.
    pub fn run(
        &self,
        hugr: &mut impl HugrMut,
        registry: &ExtensionRegistry,
    ) -> Result<(), HSeriesPassError> {
        self.lazify_measure(hugr, registry)?;
        self.force_order(hugr, registry)?;
        Ok(())
    }

    pub fn lazify_measure(&self, hugr: &mut impl HugrMut, registry: &ExtensionRegistry) -> Result<(), LazifyMeasurePassError> {
        LazifyMeasurePass::default().with_validation_level(self.validation_level).run(hugr, registry)
    }

    pub fn force_order(&self, hugr: &mut impl HugrMut, registry: &ExtensionRegistry) -> Result<(), HSeriesPassError> {
        self.validation_level
            .run_validated_pass(hugr, registry, |hugr, _| {
                force_order(hugr, hugr.root(), |hugr, node| {
                    let optype = hugr.get_optype(node);
                    if Tk2Op::try_from(optype).is_ok() || LazyQuantumOp::try_from(optype).is_ok() {
                        // quantum ops are lifted as early as possible
                        -1
                    } else if let Ok(FutureOpDef::Read) = hugr.get_optype(node).try_into() {
                        // read ops are sunk as late as possible
                        1
                    } else {
                        0
                    }
                }).map_err(HSeriesPassError::ForceOrderError)
            })
    }

    /// Returns a new `HSeriesPass` with the given [ValidationLevel].
    pub fn with_validation_level(mut self, level: ValidationLevel) -> Self {
        self.validation_level = level;
        self
    }
}

#[cfg(test)]
mod test {
    use hugr::{
        builder::{Container, DFGBuilder, Dataflow, DataflowHugr, DataflowSubContainer},
        extension::prelude::{BOOL_T, QB_T},
        ops::handle::NodeHandle,
        std_extensions::arithmetic::float_types::ConstF64,
        type_row,
        types::Signature,
        HugrView as _,
    };
    use itertools::Itertools as _;
    use petgraph::visit::{Topo, Walker as _};
    use tket2::Tk2Op;

    use crate::{extension::futures::FutureOpDef, HSeriesPass};

    #[test]
    fn hseries_pass() {
        let registry = &tket2::extension::REGISTRY;
        let (mut hugr, [call_node, h_node, f_node, rx_node]) = {
            let mut builder =
                DFGBuilder::new(Signature::new(QB_T, vec![QB_T, BOOL_T, BOOL_T])).unwrap();
            let func = builder
                .define_function("func", Signature::new_endo(type_row![]))
                .unwrap()
                .finish_with_outputs([])
                .unwrap();
            let [qb] = builder.input_wires_arr();

            // This call node has no dependencies, so it should be lifted above
            // Future Reads and sunk below quantum ops.
            let call_node = builder
                .call(func.handle(), &[], [], registry)
                .unwrap()
                .node();

            // this LoadConstant should be pushed below the quantum ops where possible
            let angle = builder.add_load_value(ConstF64::new(1.0));
            let f_node = angle.node();

            // with no dependencies, this H should be lifted to the start
            let [qb] = builder
                .add_dataflow_op(Tk2Op::H, [qb])
                .unwrap()
                .outputs_arr();
            let h_node = qb.node();

            // depending on the angle means this op can't be lifted above the float ops
            let [qb] = builder
                .add_dataflow_op(Tk2Op::RxF64, [qb, angle])
                .unwrap()
                .outputs_arr();
            let rx_node = qb.node();

            // the Measure node will be removed. A Lazy Measure and two Future
            // Reads will be added.  The Lazy Measure will be lifted and the
            // reads will be sunk.
            let [qb, measure_result] = builder
                .add_dataflow_op(Tk2Op::Measure, [qb])
                .unwrap()
                .outputs_arr();

            let hugr = builder
                .finish_hugr_with_outputs([qb, measure_result, measure_result], registry)
                .unwrap();
            (hugr, [call_node, h_node, f_node, rx_node])
        };

        HSeriesPass::default().run(&mut hugr, registry).unwrap();

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
