//! PyO3 wrapper for rewriters.

use derive_more::From;
use hugr::{hugr::views::SiblingSubgraph, HugrView, Node, SimpleReplacement};
use itertools::Itertools;
use pyo3::prelude::*;
use std::path::PathBuf;
use tket::{
    resource::ResourceScope,
    rewrite::{CircuitRewrite, ECCRewriter, Rewriter},
    Circuit,
};

use crate::{
    circuit::{PyNode, Tk2Circuit},
    matcher::{PyCombineMatchReplaceRewriter, PyMatchReplaceRewriter},
};

/// The module definition
pub fn module(py: Python<'_>) -> PyResult<Bound<'_, PyModule>> {
    let m = PyModule::new(py, "rewrite")?;
    m.add_class::<PyECCRewriter>()?;
    m.add_class::<PyCircuitRewrite>()?;
    m.add_class::<PySubcircuit>()?;
    Ok(m)
}

/// A rewrite rule for circuits.
///
/// Python equivalent of [`CircuitRewrite`].
///
/// [`CircuitRewrite`]: tket::rewrite::CircuitRewrite
#[pyclass]
#[pyo3(name = "CircuitRewrite")]
#[derive(Debug, Clone, From)]
#[repr(transparent)]
pub struct PyCircuitRewrite {
    /// Rust representation of the circuit chunks.
    pub rewrite: SimpleReplacement,
}

#[pymethods]
impl PyCircuitRewrite {
    /// Number of nodes added or removed by the rewrite.
    ///
    /// The difference between the new number of nodes minus the old. A positive
    /// number is an increase in node count, a negative number is a decrease.
    pub fn node_count_delta(&self) -> isize {
        let old_count = self.rewrite.subgraph().node_count() as isize;
        let new_count = Circuit::new(self.rewrite.replacement()).num_operations() as isize;
        new_count - old_count
    }

    /// The replacement subcircuit.
    pub fn replacement(&self) -> Tk2Circuit {
        Circuit::new(self.rewrite.replacement().to_owned()).into()
    }

    #[new]
    fn try_new(
        source_position: PySubcircuit,
        source_circ: PyRef<Tk2Circuit>,
        replacement: Tk2Circuit,
    ) -> PyResult<Self> {
        let repl = SimpleReplacement::try_new(
            source_position.0,
            source_circ.circ.hugr(),
            replacement.circ.into_hugr(),
        )
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
        Ok(Self {
            rewrite: repl.into(),
        })
    }
}

/// An enum of all rewriters exposed to the Python API.
///
/// This type is not exposed to Python, but instead corresponds to the Python
/// type union in `rewrite.py`.
#[derive(Clone, FromPyObject)]
pub enum PyRewriter {
    /// A rewriter based on circuit equivalence classes.
    ECC(PyECCRewriter),
    /// A rewriter based on a matcher and replacer.
    MatchReplace(PyMatchReplaceRewriter),
    /// A rewriter based on a combination of matchers and replacers.
    CombineMatchReplace(PyCombineMatchReplaceRewriter),
    /// A rewriter based on a list of rewriters.
    Vec(Vec<PyRewriter>),
}

impl<H: HugrView<Node = Node>> Rewriter<ResourceScope<H>> for PyRewriter {
    type Rewrite<'c>
        = CircuitRewrite
    where
        H: 'c;

    fn get_rewrites(
        &self,
        circ: &ResourceScope<H>,
        root_node: H::Node,
    ) -> Vec<CircuitRewrite<H::Node>> {
        match self {
            Self::ECC(ecc) => ecc.0.get_rewrites(circ, root_node),
            Self::MatchReplace(rewriter) => {
                Rewriter::<ResourceScope<H>>::get_rewrites(rewriter, circ, root_node)
            }
            Self::CombineMatchReplace(rewriter) => {
                Rewriter::<ResourceScope<H>>::get_rewrites(rewriter, circ, root_node)
            }
            Self::Vec(rewriters) => rewriters
                .iter()
                .flat_map(|r| r.get_rewrites(circ, root_node))
                .collect(),
        }
    }

    fn get_all_rewrites(&self, circ: &ResourceScope<H>) -> Vec<CircuitRewrite<H::Node>> {
        match self {
            Self::ECC(ecc) => ecc.0.get_all_rewrites(circ),
            Self::MatchReplace(rewriter) => {
                Rewriter::<ResourceScope<H>>::get_all_rewrites(rewriter, circ)
            }
            Self::CombineMatchReplace(rewriter) => {
                Rewriter::<ResourceScope<H>>::get_all_rewrites(rewriter, circ)
            }
            Self::Vec(rewriters) => rewriters
                .iter()
                .flat_map(|r| r.get_all_rewrites(circ))
                .collect(),
        }
    }
}

/// A subcircuit specification.
///
/// Python equivalent of [`Subcircuit`].
///
/// [`Subcircuit`]: tket::rewrite::Subcircuit
#[pyclass]
#[pyo3(name = "Subcircuit")]
#[derive(Debug, Clone, From)]
#[repr(transparent)]
pub struct PySubcircuit(SiblingSubgraph);

#[pymethods]
impl PySubcircuit {
    #[new]
    fn from_nodes(nodes: Vec<PyNode>, circ: &Tk2Circuit) -> PyResult<Self> {
        let nodes: Vec<_> = nodes.into_iter().map_into().collect();
        Ok(Self(
            SiblingSubgraph::try_from_nodes(nodes, circ.circ.hugr())
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?,
        ))
    }
}

/// A rewriter based on circuit equivalence classes.
///
/// In every equivalence class, one circuit is chosen as the representative.
/// Valid rewrites turn a non-representative circuit into its representative,
/// or a representative circuit into any of the equivalent non-representative
#[pyclass(name = "ECCRewriter")]
#[derive(Clone, From)]
pub struct PyECCRewriter(ECCRewriter);

#[pymethods]
impl PyECCRewriter {
    /// Load a precompiled ecc rewriter from a file.
    #[staticmethod]
    pub fn load_precompiled(path: PathBuf) -> PyResult<Self> {
        Ok(Self(ECCRewriter::load_binary(path).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string())
        })?))
    }

    /// Compile an ECC rewriter from a JSON file.
    #[staticmethod]
    pub fn compile_eccs(path: &str) -> PyResult<Self> {
        Ok(Self(ECCRewriter::try_from_eccs_json_file(path).map_err(
            |e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()),
        )?))
    }

    /// Returns a list of circuit rewrites that can be applied to the given
    /// Tk2Circuit.
    pub fn get_rewrites(&self, circ: &Tk2Circuit) -> Vec<PyCircuitRewrite> {
        self.0
            .get_all_rewrites(&circ.circ)
            .into_iter()
            .map(|r| match r {
                CircuitRewrite::New { .. } => unimplemented!(),
                CircuitRewrite::Old(rewrite) => SimpleReplacement::from(rewrite).into(),
            })
            .collect()
    }
}
