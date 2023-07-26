//! Subsets of `Hugr::OpType`s used for pattern matching.
//!
//! The main reason we cannot support the full HUGR set is because
//! some custom or black box optypes are not hashable.
//!
//! We currently support the minimum set of operations needed
//! for circuit pattern matching.

#[derive(Clone, Debug, Eq, Hash, PartialEq, PartialOrd, Ord)]
pub enum LeafOp {
    /// A Hadamard gate.
    H,
    /// A T gate.
    T,
    /// An S gate.
    S,
    /// A Pauli X gate.
    X,
    /// A Pauli Y gate.
    Y,
    /// A Pauli Z gate.
    Z,
    /// An adjoint T gate.
    Tadj,
    /// An adjoint S gate.
    Sadj,
    /// A controlled X gate.
    CX,
    /// A maximally entangling ZZ phase gate.
    ZZMax,
    /// A qubit measurement operation.
    Measure,
    /// A rotation of a qubit about the Pauli Z axis by an input float angle.
    RzF64,
    /// A bitwise XOR operation.
    Xor,
}

impl From<LeafOp> for hugr::ops::LeafOp {
    fn from(op: LeafOp) -> Self {
        match op {
            LeafOp::H => hugr::ops::LeafOp::H,
            LeafOp::T => hugr::ops::LeafOp::T,
            LeafOp::S => hugr::ops::LeafOp::S,
            LeafOp::X => hugr::ops::LeafOp::X,
            LeafOp::Y => hugr::ops::LeafOp::Y,
            LeafOp::Z => hugr::ops::LeafOp::Z,
            LeafOp::Tadj => hugr::ops::LeafOp::Tadj,
            LeafOp::Sadj => hugr::ops::LeafOp::Sadj,
            LeafOp::CX => hugr::ops::LeafOp::CX,
            LeafOp::ZZMax => hugr::ops::LeafOp::ZZMax,
            LeafOp::Measure => hugr::ops::LeafOp::Measure,
            LeafOp::RzF64 => hugr::ops::LeafOp::RzF64,
            LeafOp::Xor => hugr::ops::LeafOp::Xor,
        }
    }
}

impl From<hugr::ops::LeafOp> for LeafOp {
    fn from(op: hugr::ops::LeafOp) -> Self {
        match op {
            hugr::ops::LeafOp::H => LeafOp::H,
            hugr::ops::LeafOp::T => LeafOp::T,
            hugr::ops::LeafOp::S => LeafOp::S,
            hugr::ops::LeafOp::X => LeafOp::X,
            hugr::ops::LeafOp::Y => LeafOp::Y,
            hugr::ops::LeafOp::Z => LeafOp::Z,
            hugr::ops::LeafOp::Tadj => LeafOp::Tadj,
            hugr::ops::LeafOp::Sadj => LeafOp::Sadj,
            hugr::ops::LeafOp::CX => LeafOp::CX,
            hugr::ops::LeafOp::ZZMax => LeafOp::ZZMax,
            hugr::ops::LeafOp::Measure => LeafOp::Measure,
            hugr::ops::LeafOp::RzF64 => LeafOp::RzF64,
            hugr::ops::LeafOp::Xor => LeafOp::Xor,
            _ => panic!("Unsupported LeafOp"),
        }
    }
}

impl From<&hugr::ops::LeafOp> for LeafOp {
    fn from(op: &hugr::ops::LeafOp) -> Self {
        match op {
            hugr::ops::LeafOp::H => LeafOp::H,
            hugr::ops::LeafOp::T => LeafOp::T,
            hugr::ops::LeafOp::S => LeafOp::S,
            hugr::ops::LeafOp::X => LeafOp::X,
            hugr::ops::LeafOp::Y => LeafOp::Y,
            hugr::ops::LeafOp::Z => LeafOp::Z,
            hugr::ops::LeafOp::Tadj => LeafOp::Tadj,
            hugr::ops::LeafOp::Sadj => LeafOp::Sadj,
            hugr::ops::LeafOp::CX => LeafOp::CX,
            hugr::ops::LeafOp::ZZMax => LeafOp::ZZMax,
            hugr::ops::LeafOp::Measure => LeafOp::Measure,
            hugr::ops::LeafOp::RzF64 => LeafOp::RzF64,
            hugr::ops::LeafOp::Xor => LeafOp::Xor,
            _ => panic!("Unsupported LeafOp"),
        }
    }
}

#[derive(Clone, Debug, Eq, Hash, PartialEq, PartialOrd, Ord)]
pub enum OpType {
    Input,
    Output,
    LeafOp(LeafOp),
    LoadConstant,
}

impl From<hugr::ops::OpType> for OpType {
    fn from(op: hugr::ops::OpType) -> Self {
        match op {
            hugr::ops::OpType::Input(_) => OpType::Input,
            hugr::ops::OpType::Output(_) => OpType::Output,
            hugr::ops::OpType::LeafOp(op) => OpType::LeafOp(op.into()),
            hugr::ops::OpType::LoadConstant(_) => OpType::LoadConstant,
            _ => panic!("Unsupported OpType"),
        }
    }
}

impl From<&hugr::ops::OpType> for OpType {
    fn from(op: &hugr::ops::OpType) -> Self {
        match op {
            hugr::ops::OpType::Input(_) => OpType::Input,
            hugr::ops::OpType::Output(_) => OpType::Output,
            hugr::ops::OpType::LeafOp(op) => OpType::LeafOp(op.into()),
            hugr::ops::OpType::LoadConstant(_) => OpType::LoadConstant,
            _ => panic!("Unsupported OpType"),
        }
    }
}
