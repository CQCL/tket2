use derive_more::{Display, Error, From};
use hugr::algorithms::ComposablePass;
use hugr::ops::NamedOp;
use hugr::{
    builder::{BuildError, Dataflow, DataflowHugr, FunctionBuilder},
    extension::ExtensionRegistry,
    hugr::{hugrmut::HugrMut, HugrError},
    ops::{self, DataflowOpTrait, OpTag, ValidateOp},
    std_extensions::arithmetic::{float_ops::FloatOps, float_types::ConstF64},
    types::Signature,
    Direction, Hugr, HugrView, Node, PortIndex, Wire,
};
use lazy_static::lazy_static;
use std::collections::btree_map::Entry;
use std::collections::BTreeMap;
use tket2::{extension::rotation::RotationOpBuilder, Tk2Op};

use crate::extension::qsystem::{self, QSystemOp, QSystemOpBuilder};

lazy_static! {
    /// Extension registry including [crate::extension::qsystem::REGISTRY] and
    /// [tket2::extension::rotation::ROTATION_EXTENSION].
    pub static ref REGISTRY: ExtensionRegistry = {
        let mut registry = qsystem::REGISTRY.to_owned();
        registry.register(tket2::extension::rotation::ROTATION_EXTENSION.to_owned()).unwrap();
        registry
    };
}

pub(super) fn pi_mul_f64<T: Dataflow + ?Sized>(builder: &mut T, multiplier: f64) -> Wire {
    const_f64(builder, multiplier * std::f64::consts::PI)
}

fn const_f64<T: Dataflow + ?Sized>(builder: &mut T, value: f64) -> Wire {
    builder.add_load_const(ops::Const::new(ConstF64::new(value).into()))
}

/// Errors produced by lowering [Tk2Op]s.
#[derive(Debug, Display, Error, From)]
#[non_exhaustive]
pub enum LowerTk2Error {
    /// An error raised when building the circuit.
    #[display("Error when building the circuit: {_0}")]
    BuildError(BuildError),
    /// Found an unrecognised operation.
    #[display("Unrecognised operation: {} with {_1} inputs", _0.name())]
    UnknownOp(Tk2Op, usize),
    /// An error raised when replacing an operation.
    #[display("Error when replacing op: {_0}")]
    OpReplacement(HugrError),
    /// An error raised when lowering operations.
    #[display("Error when lowering ops: {_0}")]
    CircuitReplacement(hugr::algorithms::lower::LowerError),
    /// Tk2Ops were not lowered after the pass.
    #[display("Tk2Ops were not lowered: {missing_ops:?}")]
    #[from(ignore)]
    Unlowered {
        /// The list of nodes that were not lowered.
        missing_ops: Vec<Node>,
    },
    /// Non-module HUGR can't be lowered.
    #[display("HUGR root cannot have FuncDefn, has type: {}", _0.name())]
    InvalidFuncDefn(#[error(ignore)] hugr::ops::OpType),
}

/// Lower all [Tk2Op]s to [QSystemOp]s in the Hugr
/// by lazily defining and calling functions that implement the decomposition.
/// Returns the nodes that were replaced.
///
/// # Errors
/// Returns an error if the replacement fails, which could be if the root
/// operation cannot have children of type [OpTag::FuncDefn].
fn lower_ops(hugr: &mut impl HugrMut<Node = Node>) -> Result<Vec<Node>, LowerTk2Error> {
    let mut funcs: BTreeMap<Tk2Op, Node> = BTreeMap::new();

    let root_op = hugr.get_optype(hugr.root());
    if !root_op
        .validity_flags()
        .allowed_children
        .is_superset(OpTag::FuncDefn)
    {
        return Err(LowerTk2Error::InvalidFuncDefn(root_op.clone()));
    }

    let replacements: Vec<_> = hugr
        .nodes()
        .filter_map(|n| {
            let op: Tk2Op = hugr.get_optype(n).cast()?;
            Some((n, op))
        })
        .collect();

    let mut replaced_nodes = Vec::new();

    for (node, op) in replacements {
        // retrrieve or build the function
        let func_node = match funcs.entry(op) {
            Entry::Occupied(f) => *f.get(),
            Entry::Vacant(entry) => {
                let h = build_func(op)?;
                let inserted = hugr.insert_hugr(hugr.root(), h);
                entry.insert(inserted.new_root);
                inserted.new_root
            }
        };
        let call_op: hugr::ops::OpType = hugr::ops::Call::try_new(
            hugr.get_optype(func_node)
                .as_func_defn()
                .expect("should be a function definition")
                .signature
                .clone(),
            [], // no polymorphic ops functions expected.
        )
        .expect("signature should be valid")
        .into();

        let call_static_port = call_op
            .static_input_port()
            .expect("Call should have static input");

        // replace the tk2op with the function call
        hugr.replace_op(node, call_op);

        // insert an input for the Call static input
        hugr.insert_ports(node, Direction::Incoming, call_static_port.index(), 1);

        // connect the function to the call
        hugr.connect(func_node, 0, node, call_static_port);

        replaced_nodes.push(node);
    }

    Ok(replaced_nodes)
}

fn build_func(op: Tk2Op) -> Result<Hugr, LowerTk2Error> {
    let sig = op.into_extension_op().signature().into_owned();
    let sig = Signature::new(sig.input, sig.output); // ignore extension delta
                                                     // TODO check generated names are namespaced enough
    let f_name = format!("__tk2_{}", op.name().to_lowercase());
    let mut b = FunctionBuilder::new(f_name, sig)?;
    let inputs: Vec<_> = b.input_wires().collect();
    let outputs = match (op, inputs.as_slice()) {
        (Tk2Op::H, [q]) => vec![b.build_h(*q)?],
        (Tk2Op::X, [q]) => vec![b.build_x(*q)?],
        (Tk2Op::Y, [q]) => vec![b.build_y(*q)?],
        (Tk2Op::Z, [q]) => vec![b.build_z(*q)?],
        (Tk2Op::S, [q]) => vec![b.build_s(*q)?],
        (Tk2Op::Sdg, [q]) => vec![b.build_sdg(*q)?],
        (Tk2Op::T, [q]) => vec![b.build_t(*q)?],
        (Tk2Op::Tdg, [q]) => vec![b.build_tdg(*q)?],
        (Tk2Op::Measure, [q]) => b.build_measure_flip(*q)?.into(),
        (Tk2Op::QAlloc, []) => vec![b.build_qalloc()?],
        (Tk2Op::CX, [c, t]) => b.build_cx(*c, *t)?.into(),
        (Tk2Op::CY, [c, t]) => b.build_cy(*c, *t)?.into(),
        (Tk2Op::CZ, [c, t]) => b.build_cz(*c, *t)?.into(),
        (Tk2Op::Rx, [q, angle]) => {
            let float = build_to_radians(&mut b, *angle)?;
            vec![b.build_rx(*q, float)?]
        }
        (Tk2Op::Ry, [q, angle]) => {
            let float = build_to_radians(&mut b, *angle)?;
            vec![b.build_ry(*q, float)?]
        }
        (Tk2Op::Rz, [q, angle]) => {
            let float = build_to_radians(&mut b, *angle)?;
            vec![b.add_rz(*q, float)?]
        }
        (Tk2Op::CRz, [c, t, angle]) => {
            let float = build_to_radians(&mut b, *angle)?;
            b.build_crz(*c, *t, float)?.into()
        }
        (Tk2Op::Toffoli, [a, b_, c]) => b.build_toffoli(*a, *b_, *c)?.into(),
        _ => return Err(LowerTk2Error::UnknownOp(op, inputs.len())), // non-exhaustive
    };
    Ok(b.finish_hugr_with_outputs(outputs)?)
}

fn build_to_radians(b: &mut impl Dataflow, rotation: Wire) -> Result<Wire, BuildError> {
    let turns = b.add_to_halfturns(rotation)?;
    let pi = pi_mul_f64(b, 1.0);
    let float = b.add_dataflow_op(FloatOps::fmul, [turns, pi])?.out_wire(0);
    Ok(float)
}

/// Lower `Tk2Op` operations to `QSystemOp` operations.
pub fn lower_tk2_op(
    hugr: &mut impl HugrMut<Node = Node>,
) -> Result<Vec<hugr::Node>, LowerTk2Error> {
    let mut replaced_nodes = lower_direct(hugr)?;
    replaced_nodes.extend(lower_ops(hugr)?);
    Ok(replaced_nodes)
}

fn lower_direct(hugr: &mut impl HugrMut<Node = Node>) -> Result<Vec<Node>, LowerTk2Error> {
    Ok(hugr::algorithms::replace_many_ops(hugr, |op| {
        let op: Tk2Op = op.cast()?;
        Some(match op {
            Tk2Op::TryQAlloc => QSystemOp::TryQAlloc,
            Tk2Op::QFree => QSystemOp::QFree,
            Tk2Op::Reset => QSystemOp::Reset,
            Tk2Op::MeasureFree => QSystemOp::Measure,
            _ => return None,
        })
    })
    .into_iter()
    .map(|(node, _)| node)
    .collect())
}

/// Check there are no "tket2.quantum" ops left in the HUGR.
///
/// # Errors
/// Returns vector of nodes that are not lowered.
pub fn check_lowered<H: HugrView>(hugr: &H) -> Result<(), Vec<H::Node>> {
    let unlowered: Vec<H::Node> = hugr
        .nodes()
        .filter_map(|node| {
            let optype = hugr.get_optype(node);
            optype.as_extension_op().and_then(|ext| {
                (ext.def().extension_id() == &tket2::extension::TKET2_EXTENSION_ID).then_some(node)
            })
        })
        .collect();

    if unlowered.is_empty() {
        Ok(())
    } else {
        Err(unlowered)
    }
}

/// A `Hugr -> Hugr` pass that replaces [tket2::Tk2Op] nodes to
/// equivalent graphs made of [QSystemOp]s.
///
/// Invokes [lower_tk2_op]. If validation is enabled the resulting HUGR is
/// checked with [check_lowered].
#[derive(Default, Debug, Clone)]
pub struct LowerTket2ToQSystemPass;

impl ComposablePass for LowerTket2ToQSystemPass {
    type Error = LowerTk2Error;
    type Result = ();
    type Node = Node;

    fn run(&self, hugr: &mut impl HugrMut<Node = Node>) -> Result<(), LowerTk2Error> {
        lower_tk2_op(hugr)?;
        #[cfg(test)]
        check_lowered(hugr).map_err(|missing_ops| LowerTk2Error::Unlowered { missing_ops })?;
        Ok(())
    }
}

#[cfg(test)]
mod test {
    use hugr::{
        builder::{DFGBuilder, FunctionBuilder},
        extension::prelude::{bool_t, option_type, qb_t, UnwrapBuilder as _},
        type_row, HugrView,
    };
    use tket2::{extension::rotation::rotation_type, Circuit};

    use super::*;
    use rstest::rstest;

    #[test]
    fn test_lower_direct() {
        let mut b = FunctionBuilder::new("circuit", Signature::new_endo(type_row![])).unwrap();
        let [maybe_q] = b
            .add_dataflow_op(Tk2Op::TryQAlloc, [])
            .unwrap()
            .outputs_arr();
        let [q] = b.build_unwrap_sum(1, option_type(qb_t()), maybe_q).unwrap();
        let [q] = b.add_dataflow_op(Tk2Op::Reset, [q]).unwrap().outputs_arr();
        b.add_dataflow_op(Tk2Op::QFree, [q]).unwrap();
        let [maybe_q] = b
            .add_dataflow_op(Tk2Op::TryQAlloc, [])
            .unwrap()
            .outputs_arr();
        let [q] = b.build_unwrap_sum(1, option_type(qb_t()), maybe_q).unwrap();

        let [_] = b
            .add_dataflow_op(Tk2Op::MeasureFree, [q])
            .unwrap()
            .outputs_arr();
        let mut h = b
            .finish_hugr_with_outputs([])
            .unwrap_or_else(|e| panic!("{}", e));

        let lowered = lower_direct(&mut h).unwrap();
        assert_eq!(lowered.len(), 5);
        let circ = Circuit::new(&h, h.root());
        let ops: Vec<QSystemOp> = circ
            .commands()
            .filter_map(|com| com.optype().cast())
            .collect();
        assert_eq!(
            ops,
            vec![
                QSystemOp::TryQAlloc,
                QSystemOp::Measure,
                QSystemOp::TryQAlloc,
                QSystemOp::Reset,
                QSystemOp::QFree,
            ]
        );
        assert_eq!(check_lowered(&h), Ok(()));
    }

    #[rstest]
    #[case(Tk2Op::H, Some(vec![QSystemOp::PhasedX, QSystemOp::Rz]))]
    #[case(Tk2Op::X, Some(vec![QSystemOp::PhasedX]))]
    #[case(Tk2Op::Y, Some(vec![QSystemOp::PhasedX]))]
    #[case(Tk2Op::Z, Some(vec![QSystemOp::Rz]))]
    #[case(Tk2Op::S, Some(vec![QSystemOp::Rz]))]
    #[case(Tk2Op::Sdg, Some(vec![QSystemOp::Rz]))]
    #[case(Tk2Op::T, Some(vec![QSystemOp::Rz]))]
    #[case(Tk2Op::Tdg, Some(vec![QSystemOp::Rz]))]
    #[case(Tk2Op::Rx, Some(vec![QSystemOp::PhasedX]))]
    #[case(Tk2Op::Ry, Some(vec![QSystemOp::PhasedX]))]
    #[case(Tk2Op::Rz, Some(vec![QSystemOp::Rz]))]
    // multi qubit ordering is not deterministic
    #[case(Tk2Op::CX, None)]
    #[case(Tk2Op::CY, None)]
    #[case(Tk2Op::CZ, None)]
    #[case(Tk2Op::CRz, None)]
    #[case(Tk2Op::Toffoli, None)]
    // conditional doesn't fit in to commands
    #[case(Tk2Op::Measure, None)]
    #[case(Tk2Op::QAlloc, None)]
    fn test_lower(#[case] t2op: Tk2Op, #[case] qsystem_ops: Option<Vec<QSystemOp>>) {
        // build dfg with just the op

        let h = build_func(t2op).unwrap();
        let circ = Circuit::new(&h, h.root());
        let ops: Vec<QSystemOp> = circ
            .commands()
            .filter_map(|com| com.optype().cast())
            .collect();
        if let Some(qsystem_ops) = qsystem_ops {
            assert_eq!(ops, qsystem_ops);
        }

        assert_eq!(check_lowered(&h), Ok(()));
    }

    #[test]
    fn test_mixed() {
        let mut b = DFGBuilder::new(Signature::new(rotation_type(), bool_t())).unwrap();
        let [angle] = b.input_wires_arr();
        let qalloc = b.add_dataflow_op(Tk2Op::QAlloc, []).unwrap();
        let [q] = qalloc.outputs_arr();
        let [q] = b.add_dataflow_op(Tk2Op::H, [q]).unwrap().outputs_arr();
        let rx = b.add_dataflow_op(Tk2Op::Rx, [q, angle]).unwrap();
        let [q] = rx.outputs_arr();
        let [q, bool] = b
            .add_dataflow_op(Tk2Op::Measure, [q])
            .unwrap()
            .outputs_arr();
        let qfree = b.add_dataflow_op(Tk2Op::QFree, [q]).unwrap();
        b.set_order(&qalloc, &rx);
        b.set_order(&rx, &qfree);
        let mut h = b.finish_hugr_with_outputs([bool]).unwrap();

        let lowered = lower_tk2_op(&mut h).unwrap();
        assert_eq!(lowered.len(), 5);
        // dfg, input, output, alloc + (10 for unwrap), phasedx, rz, toturns, fmul, phasedx, free +
        // 5x(float + load), measure_reset, conditional, case(input, output) * 2, flip
        // (phasedx + 2*(float + load))
        assert_eq!(h.node_count(), 59);
        assert_eq!(check_lowered(&h), Ok(()));
        if let Err(e) = h.validate() {
            panic!("{}", e);
        }
    }
}
