use hugr::ops::{CustomOp, NamedOp, OpType};
use smol_str::SmolStr;

/// Matchable operations in a circuit.
#[derive(
    Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, serde::Serialize, serde::Deserialize,
)]
pub struct MatchOp {
    /// The operation identifier
    op_name: SmolStr,
    /// The encoded operation, if necessary for comparisons.
    ///
    /// This as a temporary hack for comparing parametric operations, since
    /// OpType doesn't implement Eq, Hash, or Ord.
    encoded: Option<Vec<u8>>,
}

impl From<OpType> for MatchOp {
    fn from(op: OpType) -> Self {
        let op_name = op.name();
        let encoded = encode_op(op);
        Self { op_name, encoded }
    }
}

/// Encode a unique identifier for an operation.
///
/// Avoids encoding some data if we know the operation can be uniquely
/// identified by their name.
fn encode_op(op: OpType) -> Option<Vec<u8>> {
    match op {
        OpType::Module(_) => None,
        OpType::CustomOp(op) => {
            let opaque = match op {
                CustomOp::Extension(ext_op) => ext_op.make_opaque(),
                CustomOp::Opaque(opaque) => *opaque,
            };
            let mut encoded: Vec<u8> = Vec::new();
            // Ignore irrelevant fields
            rmp_serde::encode::write(&mut encoded, opaque.extension()).ok()?;
            rmp_serde::encode::write(&mut encoded, opaque.name()).ok()?;
            rmp_serde::encode::write(&mut encoded, opaque.args()).ok()?;
            Some(encoded)
        }
        _ => rmp_serde::encode::to_vec(&op).ok(),
    }
}
