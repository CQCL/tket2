//! Optimisation passes and related utilities for circuits.

mod commutation;
use std::error::Error;

pub use commutation::{apply_greedy_commutation, PullForwardError};

pub mod chunks;
pub use chunks::CircuitChunks;

pub mod pytket;
use hugr::{
    hugr::{hugrmut::HugrMut, views::SiblingSubgraph, HugrError},
    ops::OpType,
    Hugr,
};
pub use pytket::lower_to_pytket;

pub mod tuple_unpack;
pub use tuple_unpack::find_tuple_unpack_rewrites;

// TODO use HUGR versions once they are available

/// Replace all operations in a HUGR according to a mapping.
pub fn replace_ops<S: Into<OpType>>(
    hugr: &mut impl HugrMut,
    mapping: impl Fn(&OpType) -> Option<S>,
) -> Result<(), HugrError> {
    let replacements = hugr
        .nodes()
        .filter_map(|node| {
            let new_op = mapping(hugr.get_optype(node))?;
            Some((node, new_op))
        })
        .collect::<Vec<_>>();

    for (node, new_op) in replacements {
        hugr.replace_op(node, new_op)?;
    }

    Ok(())
}

/// Lower operations in a circuit according to a mapping to a new HUGR.
pub fn lower_ops(
    hugr: &mut impl HugrMut,
    lowering: impl Fn(&OpType) -> Option<Hugr>,
) -> Result<(), Box<dyn Error>> {
    let replacements = hugr
        .nodes()
        .filter_map(|node| {
            let hugr = lowering(hugr.get_optype(node))?;
            Some((node, hugr))
        })
        .collect::<Vec<_>>();

    for (node, replacement) in replacements {
        let subcirc = SiblingSubgraph::try_from_nodes([node], hugr)?;
        let rw = subcirc.create_simple_replacement(hugr, replacement)?;
        hugr.apply_rewrite(rw)?;
    }

    Ok(())
}
