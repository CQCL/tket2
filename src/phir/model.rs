// PHIR JSON schema: https://github.com/CQCL/phir/blob/main/schema.json

use derive_more::From;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

pub(super) type Metadata = Map<String, Value>;

#[derive(Serialize, Deserialize, Debug, PartialEq)]
pub struct Data {
    #[serde(flatten)]
    pub(super) data: DataEnum,
    #[serde(default)]
    pub(super) metadata: Metadata,
}
fn default_cvar_def_data() -> String {
    "i64".to_string()
}

fn default_qvar_def_data() -> Option<String> {
    Some("qubits".to_string())
}
fn default_ffcall_cop() -> String {
    "ffcall".to_string()
}

fn default_format() -> String {
    "PHIR/JSON".to_string()
}

fn default_version() -> String {
    "0.1.0".to_string()
}

#[derive(Serialize, Deserialize, Debug, PartialEq)]
pub(super) struct CVarDefine {
    #[serde(default = "default_cvar_def_data")]
    pub(super) data_type: String,
    pub(super) variable: Sym,
    pub(super) size: Option<u64>,
}

#[derive(Serialize, Deserialize, Debug, PartialEq)]
pub(super) struct QVarDefine {
    #[serde(default = "default_qvar_def_data")]
    pub(super) data_type: Option<String>,
    pub(super) variable: Sym,
    pub(super) size: u64,
}

#[derive(Serialize, Deserialize, Debug, PartialEq)]
pub(super) struct ExportVar {
    pub(super) variables: Vec<Sym>,
    pub(super) to: Option<Vec<Sym>>,
}

#[derive(Serialize, Deserialize, Debug, PartialEq, From)]
#[serde(tag = "data")]
pub(super) enum DataEnum {
    #[serde(rename = "cvar_define")]
    CVarDefine(CVarDefine),
    #[serde(rename = "qvar_define")]
    QVarDefine(QVarDefine),
    #[serde(rename = "cvar_export")]
    ExportVar(ExportVar),
}

#[derive(Serialize, Deserialize, Debug, PartialEq)]
pub(super) struct Op {
    #[serde(flatten)]
    pub op_enum: OpEnum,
    #[serde(default)]
    pub metadata: Metadata,
}
pub type Bit = (String, u64);
pub type Sym = String;
#[derive(Serialize, Deserialize, Debug, PartialEq, From, Clone)]
#[serde(untagged)]
pub(super) enum COpArg {
    Sym(Sym),
    Int(i64),
    Bit(Bit),
    // Variadic(Vec<Arg>),
    COp(COp),
    // Other(Value),
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, From)]
#[serde(untagged)]
pub(super) enum CopReturn {
    Sym(Sym),
    Bit(Bit),
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub(super) struct COp {
    pub cop: String,
    pub args: Vec<COpArg>,
    pub returns: Option<Vec<CopReturn>>,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, From)]
#[serde(untagged)]
pub(super) enum QOpArg {
    ListBit(Vec<Bit>),
    Bit(Bit),
}

#[derive(Serialize, Deserialize, Debug, PartialEq)]
pub(super) struct QOp {
    pub qop: String,
    pub args: Vec<QOpArg>,
    pub returns: Option<Vec<Bit>>,
}

#[derive(Serialize, Deserialize, Debug, PartialEq)]
pub(super) struct FFCall {
    #[serde(default = "default_ffcall_cop")]
    pub cop: String,
    pub function: String,
    pub args: Vec<COpArg>,
    pub returns: Option<Vec<CopReturn>>,
}

#[derive(Serialize, Deserialize, Debug, PartialEq)]
pub(super) struct MOp {
    pub mop: String,
}

#[derive(Serialize, Deserialize, Debug, PartialEq, From)]
#[serde(untagged)]
pub(super) enum OpEnum {
    Qop(QOp),
    Cop(COp),
    FFCall(FFCall),
    Mop(MOp),
}

#[derive(Serialize, Deserialize, Debug, PartialEq)]
pub(super) struct Block {
    #[serde(flatten)]
    pub block_enum: BlockEnum,
    pub metadata: Option<Map<String, Value>>,
}

#[derive(Serialize, Deserialize, Debug, PartialEq)]
pub(super) struct If {
    pub(super) condition: COp,
    pub(super) true_branch: Vec<Op>,
    pub(super) false_branch: Option<Vec<Op>>,
}

#[derive(Serialize, Deserialize, Debug, PartialEq)]
pub(super) struct Seq {
    pub(super) ops: Vec<BlockElems>,
}

#[derive(Serialize, Deserialize, Debug, PartialEq, From)]
#[serde(tag = "block")]
pub(super) enum BlockEnum {
    #[serde(rename = "sequence")]
    Seq(Seq),
    #[serde(rename = "if")]
    If(If),
}

#[derive(Serialize, Deserialize, Debug, PartialEq, From)]
#[serde(untagged)]
pub(super) enum BlockElems {
    Op(Op),
    Block(Block),
}

#[derive(Serialize, Deserialize, Debug, PartialEq)]
pub(super) struct Comment {
    #[serde(rename = "//")]
    c: String,
}

#[derive(Serialize, Deserialize, Debug, PartialEq, From)]
#[serde(untagged)]
pub(super) enum OpListElems {
    Comment(Comment),
    Op(Op),
    Block(Block),
    Data(Data),
}

/// Rust encapsulation of [PHIR](https://github.com/CQCL/phir) spec.
#[derive(Serialize, Deserialize, Debug, PartialEq)]
pub struct PHIRModel {
    #[serde(default = "default_format")]
    format: String,
    #[serde(default = "default_version")]
    version: String,
    #[serde(default)]
    metadata: Metadata,
    ops: Vec<OpListElems>,
}

impl PHIRModel {
    /// Creates a new [`PHIRModel`].
    pub fn new() -> Self {
        Self {
            format: default_format(),
            version: default_version(),
            metadata: Map::new(),
            ops: vec![],
        }
    }

    /// .
    pub(super) fn append_op(&mut self, op: impl Into<OpListElems>) {
        self.ops.push(op.into());
    }

    pub(super) fn insert_op(&mut self, index: usize, op: impl Into<OpListElems>) {
        self.ops.insert(index, op.into());
    }

    /// Returns the number of ops of this [`PHIRModel`].
    pub fn num_ops(&self) -> usize {
        self.ops.len()
    }
}

impl Default for PHIRModel {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod test {
    use std::{fs::File, io::BufReader};

    use super::*;
    #[test]
    fn test_data() {
        let example = r#"
        {
            "data": "cvar_define",
            "data_type": "i64",
            "variable": "a",
            "size": 32
        }
        "#;

        let _: Data = serde_json::from_str(example).unwrap();
    }

    #[test]
    fn test_op() {
        let example = r#"
        {
            "qop": "Measure",
            "args": [["q", 0], ["q", 1]],
            "returns": [["m", 0], ["m", 1]]
        }
        "#;

        let _: Op = serde_json::from_str(example).unwrap();
    }

    #[test]
    fn test_block() {
        let example = r#"
        {
            "block": "if",
            "condition": {"cop": "==", "args": ["m", 1]},
            "true_branch": [
                {
                    "cop": "=",
                    "args": [
                        {
                            "cop": "|",
                            "args": [
                                {"cop": "^", "args": [["c", 2], "d"]},
                                {
                                    "cop": "+",
                                    "args": [
                                        {"cop": "-", "args": ["e", 2]},
                                        {"cop": "&", "args": ["f", "g"]}
                                    ]
                                }
                            ]
                        }
                    ],
                    "returns": ["a"]
                }
            ]
        }
        "#;

        let _: Block = serde_json::from_str(example).unwrap();
    }
    #[test]
    fn test_comment() {
        let example = r#"{"//": "measure q -> m;"}"#;

        let _: Comment = serde_json::from_str(example).unwrap();
    }

    #[test]
    fn test_all() {
        let reader = BufReader::new(File::open("./src/phir/test.json").unwrap());
        let p: PHIRModel = serde_json::from_reader(reader).unwrap();
        assert_eq!(p.ops.len(), 50);
        let s = serde_json::to_string(&p).unwrap();

        let p2: PHIRModel = serde_json::from_str(&s).unwrap();

        assert_eq!(p, p2);
    }
}
