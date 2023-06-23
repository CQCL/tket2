use portgraph::graph::{Direction, NodeIndex};
use pyo3::create_exception;
use pyo3::exceptions::PyException;
use pyo3::prelude::*;
use tket2::circuit::circuit::{Circuit, CircuitRewrite};
use tket2::circuit::dag::VertexProperties;
use tket2::circuit::operation::WireType;
use tket2::circuit::py_circuit::{count_pycustom, PyRewriteIter, PySubgraph};
use tket2::passes::pattern::{node_equality, Match};
use tket2::passes::{apply_greedy, CircFixedStructPattern, PatternRewriter, RewriteGenerator};
use tket_json_rs::optype::OpType;

fn _wrap_tket_conversion<F: FnOnce(Circuit) -> Circuit>(
    f: F,
) -> impl FnOnce(Py<PyAny>) -> PyResult<Py<PyAny>> {
    |c: Py<PyAny>| (f)(Circuit::_from_tket1(c)).to_tket1()
}

#[pyfunction]
fn remove_redundancies(c: Py<PyAny>) -> PyResult<Py<PyAny>> {
    _wrap_tket_conversion(|circ| tket2::passes::squash::cx_cancel_pass(circ).0)(c)
}

create_exception!(pyrs, PyValidateError, PyException);

#[pyfunction]
fn check_soundness(circ: Circuit) -> PyResult<()> {
    tket2::validate::check_soundness(&circ).map_err(|e| PyValidateError::new_err(e.to_string()))
}

#[pyfunction]
fn greedy_pattern_rewrite(
    circ: Circuit,
    pattern: Circuit,
    rewrite_fn: Py<PyAny>,
    node_match_fn: Option<Py<PyAny>>,
) -> Circuit {
    assert!(Python::with_gil(|py| rewrite_fn.as_ref(py).is_callable()));

    // gotta be a better way to do this....
    if let Some(node_match_fn) = node_match_fn {
        apply_greedy(circ, |c| {
            // todo allow spec of node comp closure too
            let pattern = CircFixedStructPattern::from_circ(
                pattern.clone(),
                |_: &_, n: NodeIndex, op: &VertexProperties| {
                    Python::with_gil(|py| {
                        node_match_fn
                            .call1(py, (n, &op.op))
                            .unwrap()
                            .extract(py)
                            .unwrap()
                    })
                },
            );
            PatternRewriter::new(pattern, |m: Match| {
                let outc: Circuit = Python::with_gil(|py| {
                    let pd = m.into_py(py);
                    rewrite_fn.call1(py, (pd,)).unwrap().extract(py).unwrap()
                });
                (outc, 0.0)
            })
            .into_rewrites(c)
            .next()
        })
        .unwrap()
        .0
    } else {
        apply_greedy(circ, |c| {
            // todo allow spec of node comp closure too
            let pattern = CircFixedStructPattern::from_circ(pattern.clone(), node_equality());
            PatternRewriter::new(pattern, |m: Match| {
                let outc: Circuit = Python::with_gil(|py| {
                    let pd = m.into_py(py);
                    rewrite_fn.call1(py, (pd,)).unwrap().extract(py).unwrap()
                });
                (outc, 0.0)
            })
            .into_rewrites(c)
            .next()
        })
        .unwrap()
        .0
    }
}

#[pyfunction]
fn greedy_iter_rewrite(circ: Circuit, it_closure: Py<PyAny>) -> Circuit {
    Python::with_gil(|py| {
        apply_greedy(circ, |c| {
            PyRewriteIter::new(it_closure.call1(py, (c.clone(),)).unwrap(), py).next()
        })
        .unwrap()
        .0
    })
}
/// A Python module implemented in Rust.
#[pymodule]
fn pyrs(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(remove_redundancies, m)?)?;
    m.add_function(wrap_pyfunction!(tket2::passes::decompose_custom_pass, m)?)?;
    m.add_function(wrap_pyfunction!(count_pycustom, m)?)?;
    m.add_function(wrap_pyfunction!(greedy_pattern_rewrite, m)?)?;
    m.add_function(wrap_pyfunction!(greedy_iter_rewrite, m)?)?;
    m.add_function(wrap_pyfunction!(check_soundness, m)?)?;
    m.add_class::<Circuit>()?;
    m.add_class::<OpType>()?;
    m.add_class::<WireType>()?;
    m.add_class::<PySubgraph>()?;
    m.add_class::<CircuitRewrite>()?;
    m.add_class::<Direction>()?;
    m.add_class::<tket2::circuit::py_circuit::PyCustom>()?;
    m.add_class::<tket2::circuit::operation::Signature>()?;
    m.add_class::<tket2::circuit::operation::Rational>()?;
    m.add_class::<tket2::circuit::operation::Quat>()?;
    m.add_class::<tket2::circuit::py_circuit::Angle>()?;
    m.add("ValidateError", _py.get_type::<PyValidateError>())?;
    Ok(())
}
