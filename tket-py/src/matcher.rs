//! Python interface for CircuitMatcher and CircuitReplacer traits.

use derive_more::derive::{From, Into};
use hugr::{
    builder::{DFGBuilder, HugrBuilder},
    hugr::views::sibling_subgraph::InvalidSubgraph,
    types::Signature,
    HugrView,
};
use pyo3::{
    exceptions::{PyKeyError, PyValueError},
    prelude::*,
    types::{PyBool, PyDict, PyNone},
    IntoPyObjectExt,
};
// Note: These imports may need to be adjusted based on the actual tket crate
// structure
use tket::{
    resource::ResourceScope,
    rewrite::{
        matcher::{CircuitMatcher, MatchContext, MatchOutcome, Update},
        replacer::{CircuitReplacer, ReplaceWithIdentity},
        CircuitRewrite, CombineMatchReplaceRewriter, MatchReplaceRewriter, Rewriter,
    },
    Circuit, Subcircuit, TketOp,
};

use crate::{
    circuit::Tk2Circuit,
    ops::PyTketOp,
    protocol::{PyImplCircuitMatcher, PyImplCircuitReplacer},
    rewrite::PyCircuitRewrite,
};

mod circuit_unit;
pub use circuit_unit::PyCircuitUnit;

mod ref_trait_impl;
use ref_trait_impl::RefTraitImpl;

/// Python wrapper for match outcomes.
///
/// # Conversion from and to Python types
///  - converts to a PyDict with a subset of "complete", "proceed" or "skip"
///    keys.
///  - converts from a PyDict with a subset of "complete", "proceed" or "skip"
///    keys, or from any non-dict Python object using its __dict__ attributes.
#[derive(Debug, Clone, From, Into)]
pub struct PyMatchOutcome {
    pub(crate) outcome: MatchOutcome<Option<PyObject>, PyObject>,
}

/// A node ID presented as a string (to remove the N: HugrNode generic).
type NodeString = String;

impl<'py> FromPyObject<'py> for PyMatchOutcome {
    fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
        let dict = try_as_dict(ob)?;
        let mut outcome = MatchOutcome::default();
        if let Some(complete) = dict.get_item("complete")? {
            outcome.complete = Some(complete.unbind());
        }
        for key in ["proceed", "skip"] {
            let Some(val) = dict.get_item(key)? else {
                continue;
            };
            // Translate to Rust
            let rust_val = if let Ok(as_bool) = val.extract::<bool>() {
                if !as_bool {
                    continue;
                }
                Update::Unchanged
            } else {
                Update::Changed(Some(val.unbind()))
            };

            // Set value
            match key {
                "proceed" => outcome.proceed = Some(rust_val),
                "skip" => outcome.skip = Some(rust_val),
                _ => unreachable!(),
            }
        }

        Ok(Self { outcome })
    }
}

impl<'py> IntoPyObject<'py> for PyMatchOutcome {
    type Target = PyDict;

    type Output = Bound<'py, PyDict>;

    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        let dict = PyDict::new(py);
        if let Some(complete) = self.outcome.complete {
            dict.set_item("complete", complete)?;
        }
        if let Some(proceed) = self.outcome.proceed {
            let pyval = match proceed {
                Update::Unchanged => PyBool::new(py, true).to_owned().into_any(),
                Update::Changed(val) => val
                    .map(|v| v.into_bound(py))
                    .unwrap_or_else(|| PyNone::get(py).to_owned().into_any()),
            };
            dict.set_item("proceed", pyval)?;
        }
        if let Some(skip) = self.outcome.skip {
            let pyval = match skip {
                Update::Unchanged => PyBool::new(py, true).to_owned().into_any(),
                Update::Changed(val) => val
                    .map(|v| v.into_bound(py))
                    .unwrap_or_else(|| PyNone::get(py).to_owned().into_any()),
            };
            dict.set_item("skip", pyval)?;
        }
        Ok(dict)
    }
}

/// Converts to a PyDict with "match_info", "subcircuit" and "op_node" keys.
pub struct PyMatchContext {
    /// The current partial match information.
    pub match_info: PyObject,
    /// The current partial match subcircuit.
    pub subcircuit: Tk2Circuit,
    /// The current operation node.
    pub op_node: NodeString,
}

impl PyMatchContext {
    /// Create a `PyMatchContext` from a `MatchContext`.
    pub fn try_from_match_context<H: HugrView>(
        context: MatchContext<'_, Option<PyObject>, H>,
        py: Python<'_>,
    ) -> PyResult<Self> {
        let match_info = context
            .match_info
            .unwrap_or_else(|| PyNone::get(py).to_owned().unbind().into_any());
        let subcircuit = if context.subcircuit.is_empty() {
            let hugr = DFGBuilder::new(Signature::new_endo(vec![]))
                .unwrap()
                .finish_hugr()
                .unwrap();
            hugr
        } else {
            context
                .subcircuit
                .try_to_subgraph(&context.circuit)
                .map_err(|e| PyValueError::new_err(format!("Invalid subgraph: {e}")))?
                .extract_subgraph(context.circuit.hugr(), "circ")
        };
        Ok(Self {
            match_info,
            subcircuit: Circuit::new(subcircuit).into(),
            op_node: format!("{:?}", context.op_node),
        })
    }
}

impl<'py> FromPyObject<'py> for PyMatchContext {
    fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
        let dict = try_as_dict(ob)?;
        let match_info = dict
            .get_item("match_info")?
            .ok_or(PyKeyError::new_err("match_info"))?
            .unbind();
        let subcircuit = dict
            .get_item("subcircuit")?
            .ok_or(PyKeyError::new_err("subcircuit"))?
            .extract()?;
        let op_node = dict
            .get_item("op_node")?
            .ok_or(PyKeyError::new_err("op_node"))?
            .extract()?;
        Ok(Self {
            match_info,
            subcircuit,
            op_node,
        })
    }
}

impl<'py> IntoPyObject<'py> for PyMatchContext {
    type Target = PyDict;
    type Output = Bound<'py, PyDict>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        PyDict::from_sequence(
            &[
                ("match_info", self.match_info),
                ("subcircuit", self.subcircuit.into_py_any(py)?),
                ("op_node", self.op_node.into_py_any(py)?),
            ]
            .into_bound_py_any(py)?,
        )
    }
}

mod dummy_matchers {
    use pyo3::exceptions::PyNotImplementedError;
    use tket::resource::CircuitUnit;

    use crate::ops::PyExtensionOp;

    use super::*;

    /// Dummy placeholder matcher for rotation gate optimization.
    #[derive(Debug, Clone)]
    #[pyclass]
    pub struct RotationMatcher;

    #[pymethods]
    impl RotationMatcher {
        /// Create a new rotation matcher.
        #[new]
        pub fn new() -> Self {
            Self
        }

        #[pyo3(name = "match_tket_op")]
        fn py_match_tket_op(
            &self,
            op: PyTketOp,
            _op_args: Vec<PyCircuitUnit>,
            _context: PyObject,
        ) -> PyResult<PyMatchOutcome> {
            let none = Python::with_gil(|py| py.None());
            if matches!(op.op, TketOp::Rx | TketOp::Ry | TketOp::Rz) {
                Ok(MatchOutcome::default().complete(none).into())
            } else {
                Ok(MatchOutcome::default().skip(Update::Unchanged).into())
            }
        }

        #[pyo3(name = "match_extension_op")]
        fn py_match_extension_op(
            &self,
            _op: PyExtensionOp,
            _inputs: Vec<PyCircuitUnit>,
            _outputs: Vec<PyCircuitUnit>,
            _context: PyObject,
        ) -> PyResult<PyMatchOutcome> {
            Err(PyNotImplementedError::new_err("match_extension_op"))
        }
    }

    impl CircuitMatcher for RotationMatcher {
        type PartialMatchInfo = ();

        type MatchInfo = ();

        fn match_tket_op<H: HugrView>(
            &self,
            op: TketOp,
            _args: &[CircuitUnit<H::Node>],
            _match_context: MatchContext<Self::PartialMatchInfo, H>,
        ) -> MatchOutcome<Self::PartialMatchInfo, Self::MatchInfo> {
            if matches!(op, TketOp::Rx | TketOp::Ry | TketOp::Rz) {
                MatchOutcome::default().complete(()).into()
            } else {
                MatchOutcome::default().skip(Update::Unchanged).into()
            }
        }
    }
}

use dummy_matchers::RotationMatcher;

/// Replace a match with the identity circuit.
#[derive(Debug, Clone)]
#[pyclass(name = "ReplaceWithIdentity")]
pub struct PyReplaceWithIdentity;

impl<M> CircuitReplacer<M> for PyReplaceWithIdentity {
    fn replace_match<H: HugrView>(
        &self,
        subcircuit: &Subcircuit<H::Node>,
        circuit: &ResourceScope<H>,
        match_info: M,
    ) -> Vec<Circuit> {
        ReplaceWithIdentity.replace_match(subcircuit, circuit, match_info)
    }
}

#[pymethods]
impl PyReplaceWithIdentity {
    /// Create a new replacer that replaces a match with the identity circuit.
    #[new]
    pub fn new() -> Self {
        Self
    }

    /// Get the identity replacement for the match.
    #[pyo3(name = "replace_match")]
    pub fn py_replace_match(&self, circuit: Tk2Circuit, _match_info: PyObject) -> Vec<Tk2Circuit> {
        vec![ReplaceWithIdentity::get_replacement_identity(
            circuit.circ.circuit_signature().into_owned(),
        )
        .unwrap()
        .into()]
    }
}

/// An enum of all circuit matchers exposed to the Python API.
///
/// This type is not exposed to Python, but instead corresponds to any Python
/// type that implements the `CircuitMatcher` protocol.
#[derive(Clone, FromPyObject)]
pub enum PyCircuitMatcherEnum {
    /// The rotation matcher.
    Rotation(RotationMatcher),
    /// A Python implementation of the CircuitMatcher protocol.
    Python(PyImplCircuitMatcher),
}

/// An enum of all circuit replacers exposed to the Python API.
///
/// This type is not exposed to Python, but instead corresponds to any Python
/// type that implements the `CircuitReplacer` protocol.
#[derive(Clone, FromPyObject)]
pub enum PyCircuitReplacerEnum {
    /// The rotation replacement.
    Rotation(PyReplaceWithIdentity),
    /// A Python implementation of the CircuitReplacer protocol.
    Python(PyImplCircuitReplacer),
}

impl PyCircuitMatcherEnum {
    /// Try to cast to a matcher with MatchInfo = ()
    fn as_unit_matcher<H: HugrView>(
        &self,
    ) -> Option<impl CircuitMatcher<PartialMatchInfo = (), MatchInfo = ()> + '_> {
        match self {
            PyCircuitMatcherEnum::Rotation(m) => Some(RefTraitImpl(m)),
            PyCircuitMatcherEnum::Python(..) => None,
        }
    }

    /// Try to cast to a matcher with MatchInfo = PyObject
    fn as_pyobject_matcher<H: HugrView>(
        &self,
    ) -> Option<impl CircuitMatcher<PartialMatchInfo = Option<PyObject>, MatchInfo = PyObject> + '_>
    {
        match self {
            PyCircuitMatcherEnum::Rotation(..) => None,
            PyCircuitMatcherEnum::Python(m) => Some(RefTraitImpl(m)),
        }
    }
}

macro_rules! impl_circuit_replacer_for_enum {
    ($match_info_type:ty) => {
        impl CircuitReplacer<$match_info_type> for PyCircuitReplacerEnum {
            fn replace_match<H: HugrView>(
                &self,
                subcircuit: &Subcircuit<H::Node>,
                circuit: &ResourceScope<H>,
                match_info: $match_info_type,
            ) -> Vec<Circuit> {
                match self {
                    PyCircuitReplacerEnum::Rotation(m) => {
                        m.replace_match(subcircuit, circuit, match_info)
                    }
                    PyCircuitReplacerEnum::Python(m) => {
                        m.replace_match(subcircuit, circuit, match_info)
                    }
                }
            }
        }
    };
}

impl_circuit_replacer_for_enum!(());
impl_circuit_replacer_for_enum!(Vec<()>);
impl_circuit_replacer_for_enum!(PyObject);
impl_circuit_replacer_for_enum!(Vec<PyObject>);

impl PyCircuitReplacerEnum {
    /// Try to cast to a replacement with MatchInfo = ()
    fn as_unit_replacement<H: HugrView>(&self) -> Option<impl CircuitReplacer<()> + '_> {
        Some(RefTraitImpl(self))
    }

    /// Try to cast to a replacement with MatchInfo = Vec<()>
    fn as_vec_unit_replacement<H: HugrView>(&self) -> Option<impl CircuitReplacer<Vec<()>> + '_> {
        Some(RefTraitImpl(self))
    }

    /// Try to cast to a replacement with MatchInfo = PyObject
    fn as_pyobject_replacement<H: HugrView>(&self) -> Option<impl CircuitReplacer<PyObject> + '_> {
        Some(RefTraitImpl(self))
    }

    /// Try to cast to a replacement with MatchInfo = Vec<PyObject>
    fn as_vec_pyobject_replacement<H: HugrView>(
        &self,
    ) -> Option<impl CircuitReplacer<Vec<PyObject>> + '_> {
        Some(RefTraitImpl(self))
    }
}

/// Python wrapper for MatchReplaceRewriter.
#[derive(Clone)]
#[pyclass(name = "MatchReplaceRewriter")]
pub struct PyMatchReplaceRewriter {
    matcher: PyCircuitMatcherEnum,
    replacement: PyCircuitReplacerEnum,
}

impl<H: HugrView<Node = hugr::Node>> Rewriter<ResourceScope<H>> for PyMatchReplaceRewriter {
    fn get_rewrites(&self, circ: &ResourceScope<H>) -> Vec<CircuitRewrite> {
        // Use the actual rewriter based on the variants
        if let Some(unit_matcher) = self.matcher.as_unit_matcher::<H>() {
            if let Some(unit_replacement) = self.replacement.as_unit_replacement::<H>() {
                let rewriter = MatchReplaceRewriter::new(unit_matcher, unit_replacement);
                return rewriter.get_rewrites(circ);
            }
        }
        if let Some(pyobject_matcher) = self.matcher.as_pyobject_matcher::<H>() {
            if let Some(pyobject_replacement) = self.replacement.as_pyobject_replacement::<H>() {
                let rewriter = MatchReplaceRewriter::new(pyobject_matcher, pyobject_replacement);
                return rewriter.get_rewrites(circ);
            }
        }
        panic!("Incompatible matcher and replacement");
    }
}

#[pymethods]
impl PyMatchReplaceRewriter {
    /// Create a new rewriter from matcher and replacement implementations.
    #[new]
    fn new(matcher: PyCircuitMatcherEnum, replacement: PyCircuitReplacerEnum) -> PyResult<Self> {
        Ok(Self {
            matcher,
            replacement,
        })
    }

    /// Get all possible rewrites for a circuit.
    fn get_rewrites(&self, circuit: &Tk2Circuit) -> Vec<PyCircuitRewrite> {
        let circuit = Circuit::new(circuit.circ.hugr());
        if circuit.subgraph() == Err(InvalidSubgraph::EmptySubgraph) {
            // No matches possible in an empty circuit
            return vec![];
        }
        let circuit = ResourceScope::from_circuit(circuit);
        <Self as Rewriter<_>>::get_rewrites(&self, &circuit)
            .into_iter()
            .map(|rw| rw.to_simple_replacement(&circuit).into())
            .collect()
    }

    fn __repr__(&self) -> String {
        "MatchReplaceRewriter(...)".to_string()
    }
}

/// Python wrapper for CombineMatchReplaceRewriter.
#[derive(Clone)]
#[pyclass(name = "CombineMatchReplaceRewriter")]
pub struct PyCombineMatchReplaceRewriter {
    matchers: Vec<PyCircuitMatcherEnum>,
    replacement: PyCircuitReplacerEnum,
}

impl<H: HugrView<Node = hugr::Node>> Rewriter<ResourceScope<H>> for PyCombineMatchReplaceRewriter {
    fn get_rewrites(&self, circ: &ResourceScope<H>) -> Vec<CircuitRewrite> {
        // Use the actual rewriter based on the variants
        if let Some(unit_matchers) = self
            .matchers
            .iter()
            .map(|m| m.as_unit_matcher::<H>())
            .collect::<Option<Vec<_>>>()
        {
            if let Some(unit_replacement) = self.replacement.as_vec_unit_replacement::<H>() {
                let rewriter = CombineMatchReplaceRewriter::new(unit_matchers, unit_replacement);
                return rewriter.get_rewrites(circ);
            }
        }
        if let Some(pyobject_matcher) = self
            .matchers
            .iter()
            .map(|m| m.as_pyobject_matcher::<H>())
            .collect::<Option<Vec<_>>>()
        {
            if let Some(pyobject_replacement) = self.replacement.as_vec_pyobject_replacement::<H>()
            {
                let rewriter =
                    CombineMatchReplaceRewriter::new(pyobject_matcher, pyobject_replacement);
                return rewriter.get_rewrites(circ);
            }
        }
        panic!("Incompatible matcher and replacement");
    }
}

#[pymethods]
impl PyCombineMatchReplaceRewriter {
    /// Create a new rewriter from matcher and replacement implementations.
    #[new]
    fn new(
        matchers: Vec<PyCircuitMatcherEnum>,
        replacement: PyCircuitReplacerEnum,
    ) -> PyResult<Self> {
        Ok(Self {
            matchers,
            replacement,
        })
    }

    /// Get all possible rewrites for a circuit.
    fn get_rewrites(&self, circuit: &Tk2Circuit) -> Vec<PyCircuitRewrite> {
        let circuit = Circuit::new(circuit.circ.hugr());
        if circuit.subgraph() == Err(InvalidSubgraph::EmptySubgraph) {
            // No matches possible in an empty circuit
            return vec![];
        }
        let circuit = ResourceScope::from_circuit(circuit);
        <Self as Rewriter<_>>::get_rewrites(&self, &circuit)
            .into_iter()
            .map(|rw| rw.to_simple_replacement(&circuit).into())
            .collect()
    }

    fn __repr__(&self) -> String {
        "CombineMatchReplaceRewriter(...)".to_string()
    }
}

fn try_as_dict<'py>(ob: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyDict>> {
    ob.extract::<Bound<'py, PyDict>>()
        .or_else(|e| {
            if ob.hasattr("__dict__")? {
                ob.getattr("__dict__")?.extract()
            } else {
                Err(e)
            }
        })
        .or_else(|e| {
            if let Ok(None) = ob.extract::<Option<usize>>() {
                Ok(PyDict::new(ob.py()))
            } else {
                Err(e)
            }
        })
}

/// Module definition for the matcher functionality.
pub fn module(py: Python<'_>) -> PyResult<Bound<'_, PyModule>> {
    let m = PyModule::new(py, "matcher")?;

    // Add the CircuitUnit class
    m.add_class::<PyCircuitUnit>()?;

    // Add the main rewriter class and supporting types
    m.add_class::<PyMatchReplaceRewriter>()?;
    m.add_class::<PyCombineMatchReplaceRewriter>()?;

    // Dummy matcher and replacement classes
    m.add_class::<RotationMatcher>()?;
    m.add_class::<PyReplaceWithIdentity>()?;

    // Add documentation about the protocols
    m.add("__doc__", "Circuit matching and replacement system.\n\nSee tket.protocol for Python protocol definitions.")?;

    Ok(m)
}
