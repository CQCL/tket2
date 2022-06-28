use std::collections::HashSet;

use crate::{
    circuit::operation::{Quat, Rational},
    graph::{
        graph::{DefaultIx, EdgeIndex, IndexType, NodeIndex, NodePort, PortIndex},
        substitute::{BoundedSubgraph, SubgraphRef},
    },
};

use super::{
    circuit::{Circuit, CircuitError, CircuitRewrite},
    operation::{AngleValue, ConstValue, Op, Param},
};
use cgmath::num_traits::CheckedDiv;
use pyo3::{
    exceptions::{PyNotImplementedError, PyStopIteration, PyZeroDivisionError},
    prelude::*,
    pyclass::CompareOp,
    types::PyType,
};

use tket_json_rs::{circuit_json::SerialCircuit, optype::OpType};

impl IntoPy<PyObject> for NodeIndex {
    fn into_py(self, py: Python<'_>) -> PyObject {
        self.index().into_py(py)
    }
}

impl<'source> FromPyObject<'source> for NodeIndex {
    fn extract(ob: &'source PyAny) -> PyResult<Self> {
        Ok(NodeIndex::new(ob.extract()?))
    }
}

impl IntoPy<PyObject> for EdgeIndex {
    fn into_py(self, py: Python<'_>) -> PyObject {
        self.index().into_py(py)
    }
}

impl<'source> FromPyObject<'source> for EdgeIndex {
    fn extract(ob: &'source PyAny) -> PyResult<Self> {
        Ok(EdgeIndex::new(ob.extract()?))
    }
}

impl IntoPy<PyObject> for NodePort {
    fn into_py(self, py: Python<'_>) -> PyObject {
        (self.node.into_py(py), self.port.into_py(py)).into_py(py)
    }
}

impl<'source> FromPyObject<'source> for NodePort {
    fn extract(ob: &'source PyAny) -> PyResult<Self> {
        let pair: (NodeIndex, PortIndex) = ob.extract()?;
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

impl IntoPy<PyObject> for Op {
    fn into_py(self, py: Python<'_>) -> PyObject {
        (&self).into_py(py)
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

    pub fn node_indices(&self) -> NodeIterator {
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

    pub fn remove_noops(&mut self) {
        let c = self.clone().remove_noop();

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
pub struct PySubgraph(BoundedSubgraph);

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

#[pymethods]
impl CircuitRewrite {
    #[new]
    pub fn py_new(subg: PySubgraph, replacement: Circuit, phase: Param) -> Self {
        Self::new(subg.0, replacement.into(), phase)
    }
}

pub struct PyRewriteIter<'py> {
    inner: Py<PyAny>,
    py: Python<'py>,
}

impl<'py> PyRewriteIter<'py> {
    pub fn new(inner: Py<PyAny>, py: Python<'py>) -> Self {
        Self { inner, py }
    }
}

impl<'py> Iterator for PyRewriteIter<'py> {
    type Item = CircuitRewrite;

    fn next(&mut self) -> Option<Self::Item> {
        match self.inner.call_method0(self.py, "__next__") {
            Ok(cr) => Some(
                cr.extract(self.py)
                    .expect("Iterator didn't return a CircuitRewrite."),
            ),
            Err(err) => {
                if err.is_instance_of::<PyStopIteration>(self.py) {
                    None
                } else {
                    panic!("{}", err);
                }
            }
        }
    }
}

#[pymethods]
impl Rational {
    #[new]
    pub fn new(num: i64, denom: i64) -> Self {
        Self(num_rational::Rational64::new(num, denom))
    }

    fn num_denom(&self) -> (i64, i64) {
        (*self.0.numer(), *self.0.denom())
    }

    fn __richcmp__(&self, other: &Self, op: CompareOp) -> PyResult<bool> {
        match op {
            CompareOp::Lt => Ok(self.0 < other.0),
            CompareOp::Le => Ok(self.0 <= other.0),
            CompareOp::Eq => Ok(self.0 == other.0),
            CompareOp::Ne => Ok(self.0 != other.0),
            CompareOp::Gt => Ok(self.0 > other.0),
            CompareOp::Ge => Ok(self.0 >= other.0),
        }
    }

    fn __add__(&self, other: &Self) -> Self {
        Self(self.0 + other.0)
    }

    fn __sub__(&self, other: &Self) -> Self {
        Self(self.0 - other.0)
    }

    fn __mul__(&self, other: &Self) -> Self {
        Self(self.0 * other.0)
    }

    fn __truediv__(&self, other: &Self) -> PyResult<Self> {
        match self.0.checked_div(&other.0) {
            Some(i) => Ok(Self(i)),
            None => Err(PyZeroDivisionError::new_err("division by zero")),
        }
    }

    fn __floordiv__(&self, other: &Self) -> PyResult<Self> {
        match self.0.checked_div(&other.0) {
            Some(i) => Ok(Self(i)),
            None => Err(PyZeroDivisionError::new_err("division by zero")),
        }
    }
}

#[pymethods]
impl Quat {
    #[new]
    pub fn new(vec: [f64; 4]) -> Self {
        let [w, x, y, z] = vec;
        Self(cgmath::Quaternion::new(w, x, y, z))
    }

    fn components(&self) -> [f64; 4] {
        let v = self.0.v;
        [self.0.s, v[0], v[1], v[2]]
    }

    fn __richcmp__(&self, other: &Self, op: CompareOp) -> PyResult<bool> {
        match op {
            CompareOp::Eq => Ok(self.0 == other.0),
            CompareOp::Ne => Ok(self.0 != other.0),
            _ => Err(PyNotImplementedError::new_err("Unsupported comparison.")),
        }
    }

    fn __add__(&self, other: &Self) -> Self {
        Self(self.0 + other.0)
    }

    fn __sub__(&self, other: &Self) -> Self {
        Self(self.0 - other.0)
    }

    fn __mul__(&self, other: &Self) -> Self {
        Self(self.0 * other.0)
    }

    fn __truediv__(&self, other: f64) -> PyResult<Self> {
        Ok(Self(self.0 / other))
    }
}

#[pyclass]
#[derive(Clone, Debug, PartialEq)]
pub struct Angle {
    float: Option<f64>,
    rational: Option<Rational>,
}

#[pymethods]
impl Angle {
    #[classmethod]
    fn float(_cls: &PyType, f: f64) -> Self {
        Self {
            float: Some(f),
            rational: None,
        }
    }

    #[classmethod]
    fn rational(_cls: &PyType, r: Rational) -> Self {
        Self {
            float: None,
            rational: Some(r),
        }
    }

    fn __richcmp__(&self, other: &Self, op: CompareOp) -> PyResult<bool> {
        match op {
            CompareOp::Eq => Ok(self == other),
            CompareOp::Ne => Ok(self != other),
            _ => Err(PyNotImplementedError::new_err("Unsupported comparison.")),
        }
    }
}

impl IntoPy<PyObject> for AngleValue {
    fn into_py(self, py: Python<'_>) -> PyObject {
        match self {
            AngleValue::F64(f) => Angle {
                float: Some(f),
                rational: None,
            },
            AngleValue::Rational(r) => Angle {
                rational: Some(r),
                float: None,
            },
        }
        .into_py(py)
    }
}

impl<'source> FromPyObject<'source> for AngleValue {
    fn extract(ob: &'source PyAny) -> PyResult<Self> {
        let angle: Angle = ob.extract()?;
        match (angle.float, angle.rational) {
            (None, None) => Err(pyo3::exceptions::PyValueError::new_err(
                "Empty angle invalid.",
            )),
            (_, Some(r)) => Ok(AngleValue::Rational(r)),
            (Some(f), _) => Ok(AngleValue::F64(f)),
        }
    }
}

impl IntoPy<PyObject> for &ConstValue {
    fn into_py(self, py: Python<'_>) -> PyObject {
        match self {
            ConstValue::Bool(x) => x.into_py(py),
            ConstValue::I32(x) => x.into_py(py),
            ConstValue::F64(x) => x.into_py(py),
            ConstValue::Angle(x) => x.into_py(py),
            ConstValue::Quat64(x) => x.into_py(py),
        }
    }
}
