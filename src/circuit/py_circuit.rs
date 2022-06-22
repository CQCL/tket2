use std::collections::HashSet;

use crate::graph::{
    graph::{DefaultIx, EdgeIndex, IndexType, NodeIndex, NodePort, PortIndex},
    substitute::{BoundedSubgraph, OpenGraph, SubgraphRef},
};

use super::{
    circuit::{Circuit, CircuitError, CircuitRewrite},
    dag::{EdgeProperties, VertexProperties},
    operation::{Op, Param, WireType},
};
use pyo3::{exceptions::PyNotImplementedError, prelude::*, pyclass::CompareOp, types::PyType};
use pythonize::{depythonize, pythonize};

use tket_json_rs::circuit_json::SerialCircuit;

#[pyclass(name = "Op")]
#[derive(Clone, Debug)]
pub enum PyOp {
    H,
    CX,
    ZZMax,
    Reset,
    Input,
    Output,
    Measure,
    Barrier,
    AngleAdd,
    AngleMul,
    AngleNeg,
    QuatMul,
    RxF64,
    RzF64,
    Rotation,
    ToRotation,
    Copy,
    Noop,
    Const,
}

impl From<PyOp> for Op {
    fn from(pyop: PyOp) -> Self {
        match pyop {
            PyOp::H => Op::H,
            PyOp::CX => Op::CX,
            PyOp::ZZMax => Op::ZZMax,
            PyOp::Reset => Op::Reset,
            PyOp::Input => Op::Input,
            PyOp::Output => Op::Output,
            PyOp::Measure => Op::Measure,
            PyOp::Barrier => Op::Barrier,
            PyOp::AngleAdd => Op::AngleAdd,
            PyOp::AngleMul => Op::AngleMul,
            PyOp::AngleNeg => Op::AngleNeg,
            PyOp::QuatMul => Op::QuatMul,
            PyOp::RxF64 => Op::RxF64,
            PyOp::RzF64 => Op::RzF64,
            PyOp::Rotation => Op::Rotation,
            PyOp::ToRotation => Op::ToRotation,
            PyOp::Noop => Op::Noop(WireType::Qubit),
            _ => panic!("Can't convert {:?} to op.", pyop),
        }
    }
}

impl From<Op> for PyOp {
    fn from(op: Op) -> Self {
        match op {
            Op::H => PyOp::H,
            Op::CX => PyOp::CX,
            Op::ZZMax => PyOp::ZZMax,
            Op::Reset => PyOp::Reset,
            Op::Input => PyOp::Input,
            Op::Output => PyOp::Output,
            Op::Measure => PyOp::Measure,
            Op::Barrier => PyOp::Barrier,
            Op::AngleAdd => PyOp::AngleAdd,
            Op::AngleMul => PyOp::AngleMul,
            Op::AngleNeg => PyOp::AngleNeg,
            Op::QuatMul => PyOp::QuatMul,
            Op::RxF64 => PyOp::RxF64,
            Op::RzF64 => PyOp::RzF64,
            Op::Rotation => PyOp::Rotation,
            Op::ToRotation => PyOp::ToRotation,
            Op::Noop(_) => PyOp::Noop,
            Op::Copy { .. } => PyOp::Copy,
            Op::Const(_) => PyOp::Const,
        }
    }
}

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

impl IntoPy<PyObject> for Op {
    fn into_py(self, py: Python<'_>) -> PyObject {
        let pyop: PyOp = self.into();
        pyop.into_py(py)
    }
}

impl<'source> FromPyObject<'source> for Op {
    fn extract(ob: &'source PyAny) -> PyResult<Self> {
        let pyop: PyOp = ob.extract()?;

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

    pub fn _from_tket1_circ(c: Py<PyAny>) -> Self {
        let ser: SerialCircuit =
            Python::with_gil(|py| depythonize(c.call_method0(py, "to_dict").unwrap().as_ref(py)))
                .unwrap();

        ser.into()
    }

    #[classmethod]
    pub fn from_tket1_circ(_cls: &PyType, c: Py<PyAny>) -> Self {
        Self::_from_tket1_circ(c)
    }

    pub fn to_tket1_circ(&self) -> PyResult<Py<PyAny>> {
        let reser: SerialCircuit = self.clone().into();
        Python::with_gil(|py| {
            let dict = pythonize(py, &reser).unwrap();
            let circ_module = PyModule::import(py, "pytket.circuit")?;

            Ok(circ_module
                .getattr("Circuit")?
                .call_method1("from_dict", (dict,))?
                .into())
        })
    }

    fn __richcmp__(&self, other: &Self, op: CompareOp) -> PyResult<bool> {
        match op {
            CompareOp::Eq => Ok(self == other),
            CompareOp::Ne => Ok(self != other),
            _ => Err(PyNotImplementedError::new_err("Unsupported comparison.")),
        }
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

#[pyclass]
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

#[pyclass]
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
    pub fn py_new(subg: PySubgraph, replacement: PyOpenCircuit, phase: Param) -> Self {
        Self::new(subg.0, replacement.0, phase)
    }
}
