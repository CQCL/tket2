use derive_more::From;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

type Metadata = Map<String, Value>;

#[derive(Serialize, Deserialize, Debug, PartialEq)]
pub struct Data {
    #[serde(flatten)]
    data: DataEnum,
    #[serde(skip_serializing_if = "Map::is_empty")]
    #[serde(default)]
    metadata: Metadata,
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
    pub(super) variable: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) size: Option<u64>,
}

#[derive(Serialize, Deserialize, Debug, PartialEq)]
pub(super) struct QVarDefine {
    #[serde(default = "default_qvar_def_data")]
    pub(super) data_type: Option<String>,
    pub(super) variable: String,
    pub(super) size: u64,
}

#[derive(Serialize, Deserialize, Debug, PartialEq)]
pub(super) struct ExportVar {
    pub(super) variables: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) to: Option<Vec<String>>,
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

#[derive(Serialize, Deserialize, Debug, PartialEq, From)]
#[serde(untagged)]
pub(super) enum Arg {
    Register(String),
    Number(u64),
    RegIndex((String, u64)),
    Variadic(Vec<Arg>),
    Other(Value),
}

#[derive(Serialize, Deserialize, Debug, PartialEq, From)]
#[serde(untagged)]
pub(super) enum CopArg {
    Arg(Arg),
    Cop(Cop),
}
#[derive(Serialize, Deserialize, Debug, PartialEq)]
pub(super) struct Op {
    #[serde(flatten)]
    pub op_enum: OpEnum,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub returns: Option<Vec<Arg>>,
    #[serde(skip_serializing_if = "Map::is_empty")]
    #[serde(default)]
    pub metadata: Metadata,
}

#[derive(Serialize, Deserialize, Debug, PartialEq)]
pub(super) struct Cop {
    pub cop: String,
    pub args: Vec<CopArg>,
}

#[derive(Serialize, Deserialize, Debug, PartialEq)]
pub(super) struct Qop {
    pub qop: String,
    pub args: Vec<Arg>,
}

#[derive(Serialize, Deserialize, Debug, PartialEq)]
pub(super) struct FFCall {
    #[serde(default = "default_ffcall_cop")]
    pub cop: String,
    pub function: String,
}

#[derive(Serialize, Deserialize, Debug, PartialEq)]
pub(super) struct Mop {
    pub mop: String,
}

#[derive(Serialize, Deserialize, Debug, PartialEq, From)]
#[serde(untagged)]
pub(super) enum OpEnum {
    Qop(Qop),
    Cop(Cop),
    FFCall(FFCall),
    Mop(Mop),
}

#[derive(Serialize, Deserialize, Debug, PartialEq)]
pub(super) struct Block {
    #[serde(flatten)]
    pub block_enum: BlockEnum,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<Map<String, Value>>,
}

#[derive(Serialize, Deserialize, Debug, PartialEq)]
pub(super) struct If {
    pub(super) condition: Cop,
    pub(super) true_branch: Vec<Op>,
    #[serde(skip_serializing_if = "Option::is_none")]
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

#[derive(Serialize, Deserialize, Debug, PartialEq)]
pub struct PHIRModel {
    #[serde(default = "default_format")]
    format: String,
    #[serde(default = "default_version")]
    version: String,
    #[serde(skip_serializing_if = "Map::is_empty")]
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
    pub(super) fn add_op(&mut self, op: impl Into<OpListElems>) {
        self.ops.push(op.into());
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
