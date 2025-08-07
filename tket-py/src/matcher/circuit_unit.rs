//! Python wrapper for OpArg.

use std::{cmp::Ordering, fmt};

use derive_more::derive::{From, Into};
use hugr::{core::HugrNode, HugrView, PortIndex};
use pyo3::{prelude::*, types::PyString};
use tket::{
    resource::{CircuitUnit, ResourceId, ResourceScope},
    rewrite::matcher::MatchContext,
    Subcircuit,
};

use crate::matcher::NodeString;

#[derive(Debug, Clone, Copy)]
pub enum CircuitUnitPos {
    Before,
    Within,
    After,
}

impl fmt::Display for CircuitUnitPos {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CircuitUnitPos::Before => write!(f, "before"),
            CircuitUnitPos::Within => write!(f, "within"),
            CircuitUnitPos::After => write!(f, "after"),
        }
    }
}

impl<'py> FromPyObject<'py> for CircuitUnitPos {
    fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
        let str = ob.extract::<String>()?;
        match str.as_str() {
            "before" => Ok(CircuitUnitPos::Before),
            "within" => Ok(CircuitUnitPos::Within),
            "after" => Ok(CircuitUnitPos::After),
            _ => Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                "Invalid CircuitUnitPos",
            )),
        }
    }
}

impl<'py> IntoPyObject<'py> for CircuitUnitPos {
    type Target = PyString;
    type Output = Bound<'py, PyString>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        let str = match self {
            CircuitUnitPos::Before => "before",
            CircuitUnitPos::Within => "within",
            CircuitUnitPos::After => "after",
        };
        Ok(PyString::new(py, str))
    }
}

/// A wrapper for OpArg, not itself exposed to Python!
///
/// Each variant has its own Python wrapper type:
///  - converting a PyOpArg to a Python object will create the appropriate
///    variant wrapper
///  - extracting a PyOpArg from a Python object will succeed if the object is
///    one of the variant wrappers.
#[pyclass(name = "CircuitUnit")]
#[derive(Debug, Clone, From, Into, Default)]
pub struct PyCircuitUnit {
    #[pyo3(get, set)]
    linear_index: Option<usize>,
    #[pyo3(get, set)]
    linear_pos: Option<CircuitUnitPos>,
    #[pyo3(get, set)]
    copyable_wire: Option<(NodeString, usize)>,
    #[pyo3(get, set)]
    constant_float: Option<f64>,
}

#[pymethods]
impl PyCircuitUnit {
    #[new]
    #[pyo3(signature=(/, linear_index = None, linear_pos = None, copyable_wire = None, constant_float = None))]
    fn new(
        linear_index: Option<usize>,
        linear_pos: Option<CircuitUnitPos>,
        copyable_wire: Option<(String, usize)>,
        constant_float: Option<f64>,
    ) -> Self {
        Self {
            linear_index,
            linear_pos,
            copyable_wire,
            constant_float,
        }
    }

    fn __repr__(&self) -> String {
        let mut args = Vec::new();
        if let Some(index) = self.linear_index {
            args.push(format!("linear_index={index}"));
        }
        if let Some(pos) = self.linear_pos {
            args.push(format!("linear_pos={pos}"));
        }
        if let Some(wire) = &self.copyable_wire {
            args.push(format!("wire={wire:?}"));
        }
        if let Some(constant) = self.constant_float {
            args.push(format!("constant={constant}"));
        }

        format!("PyCircuitUnit({})", args.join(", "))
    }
}

impl PyCircuitUnit {
    /// Create a `PyCircuitUnit` from a `CircuitUnit` and a `MatchContext`.
    pub fn with_context<N: HugrNode, M>(
        value: CircuitUnit<N>,
        context: &MatchContext<M, impl HugrView<Node = N>>,
    ) -> Self {
        match value {
            CircuitUnit::Resource(resource_id) => Self {
                linear_index: Some(resource_id.as_usize()),
                linear_pos: get_pos(
                    context.op_node,
                    &context.subcircuit,
                    resource_id,
                    &context.circuit,
                ),
                ..Self::default()
            },
            copyable_unit @ CircuitUnit::Copyable(wire) => Self {
                // TODO: constants
                copyable_wire: Some((format!("{:?}", wire.node()), wire.source().index())),
                constant_float: context.circuit.as_const_f64(copyable_unit),
                ..Self::default()
            },
        }
    }
}

fn get_pos<N: HugrNode>(
    node: N,
    subcircuit: &Subcircuit<N>,
    resource_id: ResourceId,
    circuit: &ResourceScope<impl HugrView<Node = N>>,
) -> Option<CircuitUnitPos> {
    let interval = subcircuit.get_interval(resource_id)?;
    let pos = circuit.get_position(node)?;
    match interval.position_in_interval(pos) {
        Ordering::Less => CircuitUnitPos::Before,
        Ordering::Equal => CircuitUnitPos::Within,
        Ordering::Greater => CircuitUnitPos::After,
    }
    .into()
}
