//! Python interface for CircuitMatcher and CircuitReplacer traits.

use derive_more::derive::{From, Into};
use hugr::{hugr::views::SiblingSubgraph, HugrView};
use pyo3::prelude::*;
// Note: These imports may need to be adjusted based on the actual tket crate structure
use tket::{
    rewrite::{
        matcher::{CircuitMatcher, MatchContext, MatchOutcome, OpArg},
        replacer::{CircuitReplacer, ReplaceWithIdentity},
        CircuitRewrite, MatchReplaceRewriter, Rewriter,
    },
    Circuit, TketOp,
};

use crate::{
    circuit::Tk2Circuit,
    ops::PyTketOp,
    protocol::{PyCircuitMatcherImpl, PyImplCircuitReplacer},
    rewrite::PyCircuitRewrite,
};

mod op_arg;
mod ref_trait_impl;
pub use op_arg::{PyConstF64Arg, PyOpArg, PyQubitOpAfterArg, PyQubitOpArg, PyQubitOpBeforeArg};
use ref_trait_impl::RefTraitImpl;

/// Python wrapper for match outcomes.
#[pyclass(name = "MatchOutcome")]
#[derive(Debug, Clone, From, Into)]
pub struct PyMatchOutcome {
    pub(crate) outcome: MatchOutcome<Option<PyObject>, PyObject>,
}

#[pymethods]
impl PyMatchOutcome {
    /// Create a new empty match outcome.
    #[new]
    fn new() -> Self {
        MatchOutcome::default().into()
    }

    /// Create a match outcome that stops matching.
    #[staticmethod]
    fn stop() -> Self {
        MatchOutcome::stop().into()
    }

    /// Create a match outcome that skips the current operation.
    #[staticmethod]
    fn skip(partial_match: PyObject) -> Self {
        MatchOutcome::default().skip(Some(partial_match)).into()
    }

    /// Create a match outcome that completes a match.
    #[staticmethod]
    fn complete(match_info: PyObject) -> Self {
        MatchOutcome::default().complete(match_info).into()
    }

    /// Create a match outcome that proceeds with partial matching.
    #[staticmethod]
    fn proceed(partial_match: PyObject) -> Self {
        MatchOutcome::default().proceed(Some(partial_match)).into()
    }

    fn __repr__(&self) -> String {
        format!("{:?}", self.outcome)
    }
}

mod dummy_matchers {
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
            _op_args: Vec<PyOpArg>,
            _context: PyObject,
        ) -> PyResult<PyMatchOutcome> {
            let none = Python::with_gil(|py| py.None());
            if matches!(op.op, TketOp::Rx | TketOp::Ry | TketOp::Rz) {
                Ok(MatchOutcome::default().complete(none).into())
            } else {
                Ok(MatchOutcome::default().skip(None).into())
            }
        }
    }

    impl CircuitMatcher for RotationMatcher {
        type PartialMatchInfo = ();
        type MatchInfo = ();

        fn match_tket_op(
            &self,
            op: TketOp,
            _op_args: &[OpArg],
            _match_context: MatchContext<Self::PartialMatchInfo, impl hugr::HugrView>,
        ) -> MatchOutcome<Self::PartialMatchInfo, Self::MatchInfo> {
            if matches!(op, TketOp::Rx | TketOp::Ry | TketOp::Rz) {
                MatchOutcome::default().complete(())
            } else {
                MatchOutcome::default().skip(())
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
        subgraph: &SiblingSubgraph<H::Node>,
        hugr: H,
        match_info: M,
    ) -> Vec<Circuit> {
        ReplaceWithIdentity.replace_match(subgraph, hugr, match_info)
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
pub enum PyCircuitMatcher {
    /// The rotation matcher.
    Rotation(RotationMatcher),
    /// A Python implementation of the CircuitMatcher protocol.
    Python(PyCircuitMatcherImpl),
}

/// An enum of all circuit replacers exposed to the Python API.
///
/// This type is not exposed to Python, but instead corresponds to any Python
/// type that implements the `CircuitReplacer` protocol.
#[derive(Clone, FromPyObject)]
pub enum PyCircuitReplacer {
    /// The rotation replacement.
    Rotation(PyReplaceWithIdentity),
    /// A Python implementation of the CircuitReplacer protocol.
    Python(PyImplCircuitReplacer),
}

/// Python wrapper for MatchReplaceRewriter.
#[derive(Clone)]
#[pyclass(name = "MatchReplaceRewriter")]
pub struct PyMatchReplaceRewriter {
    matcher: PyCircuitMatcher,
    replacement: PyCircuitReplacer,
}

impl PyCircuitMatcher {
    /// Try to cast to a matcher with MatchInfo = ()
    fn as_unit_matcher<H: HugrView>(
        &self,
    ) -> Option<impl CircuitMatcher<PartialMatchInfo = (), MatchInfo = ()> + '_> {
        match self {
            PyCircuitMatcher::Rotation(m) => Some(RefTraitImpl(m)),
            PyCircuitMatcher::Python(..) => None,
        }
    }

    /// Try to cast to a matcher with MatchInfo = PyObject
    fn as_pyobject_matcher<H: HugrView>(
        &self,
    ) -> Option<impl CircuitMatcher<PartialMatchInfo = Option<PyObject>, MatchInfo = PyObject> + '_>
    {
        match self {
            PyCircuitMatcher::Rotation(..) => None,
            PyCircuitMatcher::Python(m) => Some(RefTraitImpl(m)),
        }
    }
}

impl CircuitReplacer<()> for PyCircuitReplacer {
    fn replace_match<H: HugrView>(
        &self,
        subgraph: &SiblingSubgraph<H::Node>,
        hugr: H,
        match_info: (),
    ) -> Vec<Circuit> {
        match self {
            PyCircuitReplacer::Rotation(m) => m.replace_match(subgraph, hugr, match_info),
            PyCircuitReplacer::Python(m) => m.replace_match(subgraph, hugr, match_info),
        }
    }
}

impl CircuitReplacer<PyObject> for PyCircuitReplacer {
    fn replace_match<H: HugrView>(
        &self,
        subgraph: &SiblingSubgraph<H::Node>,
        hugr: H,
        match_info: PyObject,
    ) -> Vec<Circuit> {
        match self {
            PyCircuitReplacer::Rotation(m) => m.replace_match(subgraph, hugr, match_info),
            PyCircuitReplacer::Python(m) => m.replace_match(subgraph, hugr, match_info),
        }
    }
}

impl PyCircuitReplacer {
    /// Try to cast to a replacement with MatchInfo = ()
    fn as_unit_replacement<H: HugrView>(&self) -> Option<impl CircuitReplacer<()> + '_> {
        Some(RefTraitImpl(self))
    }

    /// Try to cast to a replacement with MatchInfo = PyObject
    fn as_pyobject_replacement<H: HugrView>(&self) -> Option<impl CircuitReplacer<PyObject> + '_> {
        Some(RefTraitImpl(self))
    }
}

impl<H: HugrView<Node = hugr::Node>> Rewriter<Circuit<H>> for PyMatchReplaceRewriter {
    fn get_rewrites(&self, circ: &Circuit<H>) -> Vec<CircuitRewrite> {
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
    fn new(matcher: PyCircuitMatcher, replacement: PyCircuitReplacer) -> PyResult<Self> {
        Ok(Self {
            matcher,
            replacement,
        })
    }

    /// Get all possible rewrites for a circuit.
    fn get_rewrites(&self, circuit: &Tk2Circuit) -> Vec<PyCircuitRewrite> {
        <Self as Rewriter<_>>::get_rewrites(&self, &circuit.circ)
            .into_iter()
            .map(|rw| rw.into())
            .collect()
    }

    fn __repr__(&self) -> String {
        "MatchReplaceRewriter(...)".to_string()
    }
}

/// Module definition for the matcher functionality.
pub fn module(py: Python<'_>) -> PyResult<Bound<'_, PyModule>> {
    let m = PyModule::new(py, "matcher")?;

    // Add the main rewriter class and supporting types
    m.add_class::<PyQubitOpArg>()?;
    m.add_class::<PyQubitOpBeforeArg>()?;
    m.add_class::<PyQubitOpAfterArg>()?;
    m.add_class::<PyConstF64Arg>()?;

    m.add_class::<PyMatchOutcome>()?;
    m.add_class::<PyMatchReplaceRewriter>()?;

    // Dummy matcher and replacement classes
    m.add_class::<RotationMatcher>()?;
    m.add_class::<PyReplaceWithIdentity>()?;

    // Add documentation about the protocols
    m.add("__doc__", "Circuit matching and replacement system.\n\nSee tket.protocol for Python protocol definitions.")?;

    Ok(m)
}
