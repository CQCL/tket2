use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

type Metadata = Option<Map<String, Value>>;

#[derive(Serialize, Deserialize, Debug, PartialEq)]
struct Data {
    #[serde(flatten)]
    data: DataEnum,
    #[serde(skip_serializing_if = "Option::is_none")]
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
#[serde(tag = "data")]
enum DataEnum {
    #[serde(rename = "cvar_define")]
    CVarDefine {
        #[serde(default = "default_cvar_def_data")]
        data_type: String,
        variable: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        size: Option<u64>,
    },
    #[serde(rename = "qvar_define")]
    QVarDefine {
        #[serde(default = "default_qvar_def_data")]
        data_type: Option<String>,
        variable: String,
        size: u64,
    },

    #[serde(rename = "cvar_export")]
    ExportVar {
        variables: Vec<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        to: Option<Vec<String>>,
    },
}

#[derive(Serialize, Deserialize, Debug, PartialEq)]
#[serde(untagged)]
enum Arg {
    Register(String),
    Number(u64),
    RegIndex((String, u64)),
    Variadic(Vec<Arg>),
    Other(Value),
}

#[derive(Serialize, Deserialize, Debug, PartialEq)]
#[serde(untagged)]
enum CopArg {
    Arg(Arg),
    Cop(Cop),
}
#[derive(Serialize, Deserialize, Debug, PartialEq)]
struct Op {
    #[serde(flatten)]
    op_enum: OpEnum,
    #[serde(skip_serializing_if = "Option::is_none")]
    returns: Option<Vec<Arg>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    metadata: Metadata,
}

#[derive(Serialize, Deserialize, Debug, PartialEq)]
struct Cop {
    cop: String,
    args: Vec<CopArg>,
}

#[derive(Serialize, Deserialize, Debug, PartialEq)]
#[serde(untagged)]
enum OpEnum {
    Qop {
        qop: String,
        args: Vec<Arg>,
    },
    Cop(Cop),
    FFCall {
        #[serde(default = "default_ffcall_cop")]
        cop: String,
        function: String,
    },
    Mop {
        mop: String,
    },
}

#[derive(Serialize, Deserialize, Debug, PartialEq)]
struct Block {
    #[serde(flatten)]
    block_enum: BlockEnum,
    #[serde(skip_serializing_if = "Option::is_none")]
    metadata: Option<Map<String, Value>>,
}

#[derive(Serialize, Deserialize, Debug, PartialEq)]
#[serde(tag = "block")]
enum BlockEnum {
    #[serde(rename = "sequence")]
    Seq { ops: Vec<BlockElems> },
    #[serde(rename = "if")]
    If {
        condition: Cop,
        true_branch: Vec<Op>,
        #[serde(skip_serializing_if = "Option::is_none")]
        false_branch: Option<Vec<Op>>,
    },
}

#[derive(Serialize, Deserialize, Debug, PartialEq)]
#[serde(untagged)]
enum BlockElems {
    Op(Op),
    Block(Block),
}

#[derive(Serialize, Deserialize, Debug, PartialEq)]
struct Comment {
    #[serde(rename = "//")]
    c: String,
}

#[derive(Serialize, Deserialize, Debug, PartialEq)]
#[serde(untagged)]
enum OpListElems {
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
    #[serde(skip_serializing_if = "Option::is_none")]
    metadata: Metadata,
    ops: Vec<OpListElems>,
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
