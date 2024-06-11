//! Load pre-compiled guppy functions.

use std::path::Path;
use std::{fs, io};

use hugr::ops::{NamedOp, OpTag, OpTrait, OpType};
use hugr::{Hugr, HugrView};
use itertools::Itertools;
use thiserror::Error;

use crate::{Circuit, CircuitError};

/// Loads a pre-compiled guppy file.
pub fn load_guppy_json_file(
    path: impl AsRef<Path>,
    function: &str,
) -> Result<Circuit, CircuitLoadError> {
    let file = fs::File::open(path)?;
    let reader = io::BufReader::new(file);
    load_guppy_json_reader(reader, function)
}

/// Loads a pre-compiled guppy file from a json string.
pub fn load_guppy_json_str(json: &str, function: &str) -> Result<Circuit, CircuitLoadError> {
    let reader = json.as_bytes();
    load_guppy_json_reader(reader, function)
}

/// Loads a pre-compiled guppy file from a reader.
pub fn load_guppy_json_reader(
    reader: impl io::Read,
    function: &str,
) -> Result<Circuit, CircuitLoadError> {
    let hugr: Hugr = serde_json::from_reader(reader)?;
    find_function(hugr, function)
}

/// Looks for the required function in a HUGR compiled from a guppy module.
///
/// Guppy functions are compiled into a root module, with each function as a `FuncDecl` child.
/// Each `FuncDecl` contains a `CFG` operation that defines the function.
///
/// Currently we only support functions where the CFG operation has a single `DataflowBlock` child,
/// which we use as the root of the circuit. We (currently) do not support control flow primitives.
///
/// # Errors
///
/// - If the root of the HUGR is not a module operation.
/// - If the function is not found in the module.
/// - If the function has control flow primitives.
fn find_function(hugr: Hugr, function_name: &str) -> Result<Circuit, CircuitLoadError> {
    // Find the root module.
    let module = hugr.root();
    if !OpTag::ModuleRoot.is_superset(hugr.get_optype(module).tag()) {
        return Err(CircuitLoadError::NonModuleRoot {
            root_op: hugr.get_optype(module).clone(),
        });
    }

    // Find the function declaration.
    fn func_name(op: &OpType) -> &str {
        match op {
            OpType::FuncDefn(decl) => &decl.name,
            _ => "",
        }
    }

    let Some(function) = hugr
        .children(module)
        .find(|&n| func_name(hugr.get_optype(n)) == function_name)
    else {
        let available_functions = hugr
            .children(module)
            .map(|n| func_name(hugr.get_optype(n)).to_string())
            .collect();
        return Err(CircuitLoadError::FunctionNotFound {
            function: function_name.to_string(),
            available_functions,
        });
    };

    // Find the CFG operation.
    let invalid_cfg = CircuitLoadError::InvalidControlFlow {
        function: function_name.to_string(),
    };
    let Ok(cfg) = hugr.children(function).skip(2).exactly_one() else {
        return Err(invalid_cfg);
    };

    // Find the single dataflow block to use as the root of the circuit.
    // The cfg node should only have the dataflow block and an exit node as children.
    let mut cfg_children = hugr.children(cfg);
    let Some(dataflow) = cfg_children.next() else {
        return Err(invalid_cfg);
    };
    if cfg_children.nth(1).is_some() {
        return Err(invalid_cfg);
    }

    let circ = Circuit::try_new(hugr, dataflow)?;
    Ok(circ)
}

/// Error type for conversion between `Op` and `OpType`.
#[derive(Debug, Error)]
pub enum CircuitLoadError {
    /// Cannot load the circuit file.
    #[error("Cannot load the circuit file: {0}")]
    InvalidFile(#[from] io::Error),
    /// Invalid JSON
    #[error("Invalid JSON. {0}")]
    InvalidJson(#[from] serde_json::Error),
    /// The root node is not a module operation.
    #[error(
        "Expected a HUGR with a module at the root, but found a {} instead.",
        root_op.name()
    )]
    NonModuleRoot {
        /// The root operation.
        root_op: OpType,
    },
    /// The function is not found in the module.
    #[error(
        "Function '{function}' not found in the loaded module. Available functions: [{}]",
        available_functions.join(", ")
    )]
    FunctionNotFound {
        /// The function name.
        function: String,
        /// The available functions.
        available_functions: Vec<String>,
    },
    /// The function has an invalid control flow structure.
    #[error("Function '{function}' has an invalid control flow structure. Currently only flat functions with no control flow primitives are supported.")]
    InvalidControlFlow {
        /// The function name.
        function: String,
    },
    /// Error loading the circuit.
    #[error("Error loading the circuit: {0}")]
    CircuitLoadError(#[from] CircuitError),
}
