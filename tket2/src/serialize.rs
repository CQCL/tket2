//! Utilities for serializing circuits.
//!
//! See [`crate::serialize::pytket`] for serialization to and from the legacy pytket format.
pub mod pytket;

use std::io;

use hugr::extension::ExtensionRegistryError;
use hugr::hugr::ValidationError;
pub use pytket::{
    load_tk1_json_file, load_tk1_json_reader, load_tk1_json_str, save_tk1_json_file,
    save_tk1_json_str, save_tk1_json_writer, TKETDecode,
};

use derive_more::{Display, Error, From};
use hugr::ops::{NamedOp, OpTag, OpTrait, OpType};
use hugr::package::{Package, PackageEncodingError, PackageError, PackageValidationError};
use hugr::{Hugr, HugrView, Node};

use crate::extension::REGISTRY;
use crate::{Circuit, CircuitError};

/// An encoded path pointing to a node in the HUGR,
/// to be used as the [`Circuit`] root.
///
/// This key should not be used in the in-memory structure, as any modifications to the HUGR may
/// invalidate the path.
///
/// TODO: Implement the path pointer. Currently this entry is not used.
#[allow(unused)]
const METADATA_ENTRYPOINT: &str = "TKET2.entrypoint";

impl<T: AsRef<Hugr>> Circuit<T> {
    /// Store the circuit as a HUGR in json format.
    ///
    /// # Errors
    ///
    /// - If the circuit's parent is not the root of the HUGR. This will be relaxed in the future.
    ///
    // TODO: Store the path pointer on `METADATA_ENTRYPOINT`.
    // We may need mutable access to `T` to avoid cloning the HUGR to add the entry.
    pub fn to_hugr_writer(&self, writer: impl io::Write) -> Result<(), CircuitStoreError> {
        let hugr = self.hugr().as_ref();

        if self.parent() != hugr.root() {
            return Err(CircuitStoreError::NonRootCircuit {
                parent: self.parent(),
            });
        }

        Ok(serde_json::to_writer(writer, hugr)?)
    }

    /// Store the circuit as a package in json format.
    ///
    /// If the circuit is not a function in a module-rooted HUGR, a new module
    /// is created with a `main` function containing it.
    ///
    /// # Errors
    ///
    /// - If the circuit's parent is not the root of the HUGR or a function in
    ///   the root's module.
    ///
    // TODO: Store the path pointer on `METADATA_ENTRYPOINT` instead.
    pub fn to_package_writer(&self, writer: impl io::Write) -> Result<(), CircuitStoreError> {
        let hugr = self.hugr().as_ref();

        // Check if we support storing the circuit as a package.
        //
        // This restriction may be relaxed once `METADATA_ENTRYPOINT` is implemented.
        let circuit_is_at_root = self.parent() == hugr.root();
        let circuit_is_fn_at_module_root = OpTag::ModuleRoot
            .is_superset(hugr.get_optype(hugr.root()).tag())
            && hugr.get_parent(self.parent()) == Some(hugr.root());
        if !circuit_is_at_root && !circuit_is_fn_at_module_root {
            return Err(CircuitStoreError::NonRootCircuit {
                parent: self.parent(),
            });
        }

        let mut pkg = Package::from_hugr(hugr.clone())?;
        pkg.extensions = self.required_extensions().to_owned();

        Ok(pkg.to_json_writer(writer)?)
    }
}

impl Circuit<Hugr> {
    /// Load a circuit from a package or hugr json.
    ///
    /// If the json encodes a package, one of the modules must contain a
    /// function called `function_name`.
    ///
    /// Otherwise, the json must encode a module-rooted HUGR containing the
    /// named function.
    pub fn load_function_reader(
        json: impl io::Read,
        function_name: impl AsRef<str>,
    ) -> Result<Self, CircuitLoadError> {
        let mut pkg = Package::from_json_reader(json)?;
        pkg.update_validate(&mut REGISTRY.clone())?;
        let Package {
            modules,
            extensions,
        } = pkg;

        let (_module_idx, mut circ) = find_function_in_modules(modules, function_name.as_ref())?;
        if !extensions.is_empty() {
            circ.set_required_extensions(extensions);
        }
        Ok(circ)
    }

    /// Load a circuit from a hugr json.
    ///
    /// The circuit points to the Hugr's root or, if the `TKET2.entrypoint` metadata is present,
    /// the indicated node.
    ///
    /// # Errors
    ///
    /// - If the target circuit root is not a dataflow container.
    pub fn load_hugr_reader(json: impl io::Read) -> Result<Self, CircuitLoadError> {
        let mut hugr: Hugr = serde_json::from_reader(json)?;
        hugr.update_validate(&REGISTRY.clone())?;
        // TODO: Read the entrypoint from the metadata.
        let root = hugr.root();
        Ok(Circuit::try_new(hugr, root)?)
    }
}

/// Error type for serialization operations on [`Circuit`]s.
#[derive(Debug, Display, Error, From)]
#[non_exhaustive]
pub enum CircuitStoreError {
    /// Could not encode the hugr json.
    EncodingError(serde_json::Error),
    /// Cannot load the circuit file.
    #[display("Cannot write to the circuit file: {_0}")]
    InvalidFile(io::Error),
    /// The circuit could not be stored as a package.
    PackageStore(PackageError),
    /// The circuit's parent is not the root of the HUGR.
    #[display("The circuit's parent {parent} is not the root of the HUGR.")]
    NonRootCircuit {
        /// The parent node.
        parent: Node,
    },
}

/// Error type for deserialization operations on [`Circuit`]s.
#[derive(Debug, Display, Error, From)]
#[non_exhaustive]
pub enum CircuitLoadError {
    /// Cannot load the circuit file.
    #[display("Cannot load the circuit file: {_0}")]
    #[from]
    InvalidFile(io::Error),
    /// Invalid JSON
    #[display("Invalid HUGR JSON. {_0}")]
    #[from]
    InvalidJson(serde_json::Error),
    /// The root node is not a module operation.
    #[display(
        "Expected a HUGR with a module at the root, but found a {} instead.",
        root_op.name()
    )]
    NonModuleRoot {
        /// The root operation.
        root_op: OpType,
    },
    /// The function is not found in the module.
    #[display(
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
    #[display("Function '{function}' has an invalid control flow structure. Currently only flat functions with no control flow primitives are supported.")]
    InvalidControlFlow {
        /// The function name.
        function: String,
    },
    /// Error loading the circuit.
    #[display("Error loading the circuit: {_0}")]
    #[from]
    CircuitLoadError(CircuitError),
    /// Error loading the circuit.
    #[from]
    PackageError(PackageEncodingError),

    /// Error validating the loaded circuit.
    #[from]
    ValidationError(ValidationError),
    /// An error that can occur in defining an extension registry while loading the circuit.
    #[from]
    ExtensionError(ExtensionRegistryError),
    /// The encoded HUGR package must have a single HUGR.
    #[display("The encoded HUGR package must have a single HUGR, but it has {count} HUGRs.")]
    InvalidNumHugrs {
        /// The number of HUGRs encountered in the encoded HUGR package.
        count: usize,
    },
}

impl From<PackageEncodingError> for CircuitStoreError {
    fn from(e: PackageEncodingError) -> Self {
        match e {
            PackageEncodingError::Package(e) => CircuitStoreError::PackageStore(e),
            PackageEncodingError::JsonEncoding(e) => CircuitStoreError::EncodingError(e),
            PackageEncodingError::IOError(e) => CircuitStoreError::InvalidFile(e),
            _ => panic!("Unexpected package encoding error: {e}"),
        }
    }
}

impl From<PackageValidationError> for CircuitLoadError {
    fn from(e: PackageValidationError) -> Self {
        match e {
            PackageValidationError::Validation(e) => CircuitLoadError::ValidationError(e),
            PackageValidationError::Extension(e) => CircuitLoadError::ExtensionError(e),
            _ => panic!("Unexpected package validation error: {e}"),
        }
    }
}

/// Looks for the circuit entrypoint in a list of modules, and returns a new
/// circuit pointing to it.
///
/// The modules are searched in order, and the first match is returned.
///
/// # Errors
///
/// - If any of the HUGR roots is not a module operation.
/// - If the function is not found in any module.
fn find_function_in_modules(
    modules: impl IntoIterator<Item = Hugr>,
    function_name: &str,
) -> Result<(usize, Circuit), CircuitLoadError> {
    let mut available_functions = Vec::new();
    for (i, hugr) in modules.into_iter().enumerate() {
        match find_function(hugr, function_name) {
            Ok(circ) => return Ok((i, circ)),
            Err(CircuitLoadError::FunctionNotFound {
                available_functions: fns,
                ..
            }) => {
                available_functions.extend(fns);
                continue;
            }
            Err(e) => return Err(e),
        }
    }
    Err(CircuitLoadError::FunctionNotFound {
        function: function_name.to_string(),
        available_functions,
    })
}

/// Looks for the circuit entrypoint in a HUGR, and returns a new circuit pointing to it.
///
/// # Errors
///
/// - If the root of the HUGR is not a module operation.
/// - If the function is not found in the module.
fn find_function(hugr: Hugr, function_name: &str) -> Result<Circuit, CircuitLoadError> {
    // Find the root module.
    let module = hugr.root();
    if !OpTag::ModuleRoot.is_superset(hugr.get_optype(module).tag()) {
        return Err(CircuitLoadError::NonModuleRoot {
            root_op: hugr.get_optype(module).clone(),
        });
    }

    // Find the function definition.
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

    let circ = Circuit::try_new(hugr, function)?;
    Ok(circ)
}

#[cfg(test)]
mod tests {
    use crate::Tk2Op;

    use super::*;
    use std::io::Cursor;

    use cool_asserts::assert_matches;
    use hugr::builder::{
        Container, DFGBuilder, Dataflow, DataflowHugr, DataflowSubContainer, HugrBuilder,
        ModuleBuilder,
    };
    use hugr::extension::prelude::QB_T;
    use hugr::ops::handle::NodeHandle;
    use hugr::type_row;
    use hugr::types::Signature;
    use itertools::Itertools;
    use rstest::{fixture, rstest};

    /// A circuit based on a DFG-rooted HUGR.
    #[fixture]
    fn root_circ() -> Circuit {
        let mut h = DFGBuilder::new(Signature::new(vec![], vec![QB_T])).unwrap();

        let res = h.add_dataflow_op(Tk2Op::QAlloc, []).unwrap();
        let q = res.out_wire(0);

        h.finish_hugr_with_outputs([q], &REGISTRY).unwrap().into()
    }

    #[fixture]
    fn function_circ() -> Circuit {
        let mut h = ModuleBuilder::new();

        let mut f = h
            .define_function("banana", Signature::new(vec![], vec![QB_T]))
            .unwrap();
        let res = f.add_dataflow_op(Tk2Op::QAlloc, []).unwrap();
        let q = res.out_wire(0);
        let func_node = f.finish_with_outputs([q]).unwrap().handle().node();

        Circuit::new(h.finish_hugr(&REGISTRY).unwrap(), func_node)
    }

    /// A circuit located inside a function in a module.
    #[fixture]
    fn nested_circ() -> Circuit {
        let mut h = ModuleBuilder::new();

        let mut f = h
            .define_function("banana", Signature::new(vec![], vec![QB_T]))
            .unwrap();
        let dfg = {
            let mut dfg = f
                .dfg_builder(Signature::new(vec![], type_row![QB_T]), [])
                .unwrap();
            let res = dfg.add_dataflow_op(Tk2Op::QAlloc, []).unwrap();
            let q = res.out_wire(0);
            dfg.finish_with_outputs([q]).unwrap()
        };
        f.finish_with_outputs(dfg.outputs())
            .unwrap()
            .handle()
            .node();

        Circuit::new(h.finish_hugr(&REGISTRY).unwrap(), dfg.node())
    }

    #[fixture]
    fn multi_module_pkg() -> Package {
        fn define(name: &str, h: &mut ModuleBuilder<Hugr>) -> Node {
            let f = h
                .define_function(name, Signature::new(vec![QB_T], vec![QB_T]))
                .unwrap();
            let inputs = f.input_wires().collect_vec();
            f.finish_with_outputs(inputs).unwrap().handle().node()
        }

        let mut mod1 = ModuleBuilder::new();
        define("apple", &mut mod1);
        define("banana", &mut mod1);
        let mod1 = mod1.finish_prelude_hugr().unwrap();

        let mut mod2 = ModuleBuilder::new();
        define("foo", &mut mod2);
        define("bar", &mut mod2);
        define("banana", &mut mod2);
        let mod2 = mod2.finish_prelude_hugr().unwrap();

        Package::new([mod1, mod2], []).unwrap()
    }

    /// Test roundtrips of a circuit with a root parent.
    #[rstest]
    fn root_circuit_store(root_circ: Circuit) {
        let mut buf = Vec::new();
        root_circ.to_hugr_writer(&mut buf).unwrap();
        let circ2 = Circuit::load_hugr_reader(Cursor::new(buf)).unwrap();
        assert_eq!(root_circ, circ2);

        let mut buf = Vec::new();
        root_circ.to_package_writer(&mut buf).unwrap();
        let circ2 = Circuit::load_function_reader(Cursor::new(buf), "main").unwrap();
        let extracted_circ2 = circ2.extract_dfg().unwrap();

        assert_eq!(root_circ, extracted_circ2);
    }

    #[rstest]
    fn func_circuit_store(function_circ: Circuit) {
        let mut buf = Vec::new();
        function_circ.to_package_writer(&mut buf).unwrap();
        let circ2 = Circuit::load_function_reader(Cursor::new(buf), "banana").unwrap();

        assert_eq!(function_circ, circ2);
    }

    #[rstest]
    fn serialize_package_errors(multi_module_pkg: Package) {
        let pkg_json = multi_module_pkg.to_json().unwrap();

        match Circuit::load_function_reader(Cursor::new(&pkg_json), "not_found") {
            Err(CircuitLoadError::FunctionNotFound {
                function,
                available_functions,
            }) => {
                assert_eq!(function, "not_found");
                assert_eq!(
                    available_functions,
                    ["apple", "banana", "foo", "bar", "banana"]
                );
            }
            Err(e) => panic!("Expected FunctionNotFound error got {e}."),
            Ok(_) => panic!("Expected an error."),
        };

        assert_matches!(
            Circuit::load_function_reader(Cursor::new(&pkg_json), "banana"),
            Ok(_)
        )
    }

    #[rstest]
    fn root_errors(function_circ: Circuit, nested_circ: Circuit) {
        // Trying to store a non-root circuit as a hugr.
        let mut buf = Vec::new();
        assert_matches!(
            function_circ.to_hugr_writer(&mut buf),
            Err(CircuitStoreError::NonRootCircuit { .. })
        );

        // Trying to store a non-root (and non-function-in-a-module) circuit as a package.
        let mut buf = Vec::new();
        assert_matches!(
            nested_circ.to_package_writer(&mut buf),
            Err(CircuitStoreError::NonRootCircuit { .. })
        );
    }
}
