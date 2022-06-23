use std::collections::HashSet;

use crate::graph::{
    graph::{DefaultIx, EdgeIndex, IndexType, NodeIndex, NodePort, PortIndex},
    substitute::{BoundedSubgraph, OpenGraph, SubgraphRef},
};

use super::{
    circuit::{Circuit, CircuitError, CircuitRewrite},
    dag::{EdgeProperties, VertexProperties},
    operation::{Op, Param},
};
use pyo3::{exceptions::PyNotImplementedError, prelude::*, pyclass::CompareOp, types::PyType};

use tket_json_rs::{circuit_json::SerialCircuit, optype::OpType};

impl<Ix: IndexType> IntoPy<PyObject> for NodeIndex<Ix> {
    fn into_py(self, py: Python<'_>) -> PyObject {
        self.index().into_py(py)
    }
}

impl<'source, Ix: IndexType> FromPyObject<'source> for NodeIndex<Ix> {
    fn extract(ob: &'source PyAny) -> PyResult<Self> {
        Ok(NodeIndex::new(ob.extract()?))
    }
}

impl<Ix: IndexType> IntoPy<PyObject> for EdgeIndex<Ix> {
    fn into_py(self, py: Python<'_>) -> PyObject {
        self.index().into_py(py)
    }
}

impl<'source, Ix: IndexType> FromPyObject<'source> for EdgeIndex<Ix> {
    fn extract(ob: &'source PyAny) -> PyResult<Self> {
        Ok(EdgeIndex::new(ob.extract()?))
    }
}

impl<Ix: IndexType> IntoPy<PyObject> for NodePort<Ix> {
    fn into_py(self, py: Python<'_>) -> PyObject {
        (self.node.into_py(py), self.port.into_py(py)).into_py(py)
    }
}

impl<'source, Ix: IndexType> FromPyObject<'source> for NodePort<Ix> {
    fn extract(ob: &'source PyAny) -> PyResult<Self> {
        let pair: (NodeIndex<Ix>, PortIndex) = ob.extract()?;
        Ok(NodePort::new(pair.0, pair.1))
    }
}

impl IntoPy<PyObject> for PortIndex {
    fn into_py(self, py: Python<'_>) -> PyObject {
        self.index().into_py(py)
    }
}

impl<'source> FromPyObject<'source> for PortIndex {
    fn extract(ob: &'source PyAny) -> PyResult<Self> {
        Ok(PortIndex::new(ob.extract()?))
    }
}

impl IntoPy<PyObject> for &Op {
    fn into_py(self, py: Python<'_>) -> PyObject {
        let pyop: OpType = self.into();
        pyop.into_py(py)
    }
}

impl<'source> FromPyObject<'source> for Op {
    fn extract(ob: &'source PyAny) -> PyResult<Self> {
        let pyop: OpType = ob.extract()?;

        Ok(pyop.into())
    }
}

impl std::convert::From<CircuitError> for PyErr {
    fn from(s: CircuitError) -> Self {
        pyo3::exceptions::PyRuntimeError::new_err(s.0)
    }
}

#[pymethods]
impl Circuit {
    #[new]
    pub fn py_new() -> Self {
        Self::new()
    }
    // pub fn py_apply_rewrite(&mut self, rewrite: CircuitRewrite) {
    //     self.dag.apply_rewrite(rewrite.graph_rewrite).unwrap();
    //     self.phase += rewrite.phase;
    // }

    #[pyo3(name = "boundary")]
    pub fn py_boundary(&self) -> [usize; 2] {
        let [i, o] = self.boundary();
        [i.index(), o.index()]
    }

    pub fn nodes(&self) -> NodeIterator {
        // TODO find a way to do this without the collect
        // or just return the Vec
        NodeIterator(
            self.dag
                .node_indices()
                .collect::<Vec<NodeIndex>>()
                .into_iter(),
        )
    }

    pub fn _from_tket1(c: Py<PyAny>) -> Self {
        let ser = SerialCircuit::_from_tket1(c);
        ser.into()
    }

    #[classmethod]
    pub fn from_tket1(_cls: &PyType, c: Py<PyAny>) -> Self {
        Self::_from_tket1(c)
    }

    pub fn to_tket1(&self) -> PyResult<Py<PyAny>> {
        let reser: SerialCircuit = self.clone().into();
        reser.to_tket1()
    }

    fn __richcmp__(&self, other: &Self, op: CompareOp) -> PyResult<bool> {
        match op {
            CompareOp::Eq => Ok(self == other),
            CompareOp::Ne => Ok(self != other),
            _ => Err(PyNotImplementedError::new_err("Unsupported comparison.")),
        }
    }

    pub fn defrag(&mut self) {
        let c = self.clone().remove_invalid();

        *self = c;
    }
}
#[pyclass]
pub struct NodeIterator(std::vec::IntoIter<NodeIndex>);
#[pymethods]
impl NodeIterator {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(mut slf: PyRefMut<'_, Self>) -> Option<NodeIndex> {
        slf.0.next()
    }
}

#[pyclass(name = "Subgraph")]
#[derive(Clone)]
pub struct PySubgraph(BoundedSubgraph<DefaultIx>);

#[pymethods]
impl PySubgraph {
    #[new]
    pub fn new(
        subg_nodes: HashSet<NodeIndex>,
        in_edges: Vec<EdgeIndex>,
        out_edges: Vec<EdgeIndex>,
    ) -> Self {
        Self(BoundedSubgraph::new(
            SubgraphRef::new(subg_nodes),
            [in_edges, out_edges],
        ))
    }
}

#[pyclass(name = "OpenCircuit")]
#[derive(Clone)]
pub struct PyOpenCircuit(OpenGraph<VertexProperties, EdgeProperties, DefaultIx>);

#[pymethods]
impl PyOpenCircuit {
    #[new]
    pub fn new(circ: Circuit) -> Self {
        Self(circ.into())
    }
}
#[pymethods]
impl CircuitRewrite {
    #[new]
    pub fn py_new(subg: PySubgraph, replacement: Circuit, phase: Param) -> Self {
        Self::new(subg.0, replacement.into(), phase)
    }
}
