use crate::{circuit::operation::Param, optype::OpType};
use serde::{Deserialize, Serialize};

#[derive(Deserialize, Serialize, Clone, Debug, PartialEq)]
pub struct Register(pub String, pub Vec<i64>);

#[derive(Deserialize, Serialize, Clone, Debug, PartialEq)]
pub struct CompositeGate {
    // List of Symbols
    args: Vec<String>,
    definition: Box<SerialCircuit>,
    name: String,
}

#[derive(Deserialize, Serialize, Clone, Debug, PartialEq)]
pub struct BoxID(uuid::Uuid);

/// Box for an operation, the enum variant names come from the names
/// of the C++ operations and are renamed if the string corresponding
/// to the operation is differently named when serializing.
#[derive(Deserialize, Serialize, Clone, Debug, PartialEq)]
#[serde(tag = "type")]
pub enum OpBox {
    CircBox {
        id: BoxID,
        circuit: SerialCircuit,
    },
    Unitary1qBox {
        id: BoxID,
        // 2x2 matrix of complex numbers
        matrix: [[(f32, f32); 2]; 2],
    },
    Unitary2qBox {
        id: BoxID,
        // 4x4 matrix of complex numbers
        matrix: [[(f32, f32); 4]; 4],
    },
    ExpBox {
        id: BoxID,
        // 4x4 matrix of complex numbers
        matrix: [[(f32, f32); 4]; 4],
        phase: f64,
    },
    PauliExpBox {
        id: BoxID,
        paulis: Vec<String>,
        // Symengine Expr
        phase: String,
    },
    PhasePolyBox {
        id: BoxID,
        n_qubits: u32,
        qubit_indices: Vec<(u32, u32)>,
    },
    Composite {
        id: BoxID,
        gate: CompositeGate,
        // Vec of Symengine Expr
        params: Vec<String>,
    },
    QControlBox {
        id: BoxID,
        n_controls: u32,
        op: Box<Operation>,
    },
}

#[derive(Deserialize, Serialize, Clone, Debug, PartialEq)]
pub struct Conditional {
    op: Box<Operation>,
    width: u32,
    value: u32,
}

#[derive(Deserialize, Serialize, Clone, Debug, PartialEq)]
pub struct Operation {
    #[serde(rename = "type")]
    pub op_type: OpType,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub n_qb: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub params: Option<Vec<Param>>,
    #[serde(rename = "box")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub op_box: Option<OpBox>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub signature: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub conditional: Option<Conditional>,
}

#[derive(Deserialize, Serialize, Clone, Debug, PartialEq)]
pub struct Command {
    pub op: Operation,
    pub args: Vec<Register>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub opgroup: Option<String>,
}

#[derive(Deserialize, Serialize, Clone, Debug, PartialEq)]
pub struct Permutation(pub Register, pub Register);

/// Pytket canonical circuit
#[derive(Deserialize, Serialize, Clone, Debug, PartialEq)]
pub struct SerialCircuit {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    // Symengine Expr
    pub phase: Param,
    pub commands: Vec<Command>,
    pub qubits: Vec<Register>,
    pub bits: Vec<Register>,
    pub implicit_permutation: Vec<Permutation>,
}
