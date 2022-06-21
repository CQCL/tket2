use crate::graph::graph::NodeIndex;

use super::{
    circuit::{Circuit, CircuitRewrite, UnitID},
    operation::{Op, WireType},
};
use crate::graph::dot::dot_string;
use pyo3::{prelude::*, types::PyType};
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

#[pymethods]
impl Circuit {
    #[new]
    pub fn py_new() -> Self {
        Self::new()
    }
    pub fn py_apply_rewrite(&mut self, rewrite: CircuitRewrite) {
        self.dag.apply_rewrite(rewrite.graph_rewrite).unwrap();
        self.phase += rewrite.phase;
    }

    pub fn py_add_vertex(&mut self, op: PyOp) -> usize {
        self.add_vertex(op.into()).index()
    }

    pub fn py_add_qid(&mut self, s: &str) -> (usize, usize) {
        let np = self.add_unitid(UnitID::Qubit {
            name: s.to_string(),
            index: vec![],
        });
        (np.node.index(), np.port.index())
    }

    pub fn py_add_edge(
        &mut self,
        source: (usize, u8),
        target: (usize, u8),
        edge_type: WireType,
    ) -> usize {
        self.add_edge(
            (NodeIndex::new(source.0), source.1),
            (NodeIndex::new(target.0), target.1),
            edge_type,
        )
        .index()
    }

    pub fn py_boundary(&self) -> [usize; 2] {
        let [i, o] = self.boundary();
        [i.index(), o.index()]
    }

    pub fn nodes(&self) -> NodeIterator {
        // TODO find a way to do this without the collect
        // or just return the Vec
        NodeIterator {
            inner: self.dag.nodes().collect::<Vec<NodeIndex>>().into_iter(),
        }
    }

    pub fn node_weight(&self, n: usize) -> Option<PyOp> {
        self.dag
            .node_weight(NodeIndex::new(n))
            .map(|vp| vp.op.clone().into())
    }

    pub fn dot_string(&self) -> String {
        dot_string(self.dag_ref())
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
}
#[pyclass]
pub struct NodeIterator {
    inner: std::vec::IntoIter<NodeIndex>,
}

#[pymethods]
impl NodeIterator {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(mut slf: PyRefMut<'_, Self>) -> Option<usize> {
        slf.inner.next().map(|n| n.index())
    }
}
