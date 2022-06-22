use pyo3::prelude::*;
use tket_rs::circuit::circuit::{Circuit, CircuitRewrite};
use tket_rs::circuit::operation::WireType;
use tket_rs::circuit::py_circuit::{PyOp, PyOpenCircuit, PySubgraph};
use tket_rs::passes::pattern::node_equality;
use tket_rs::passes::{apply_greedy, pattern_rewriter, CircFixedStructPattern};

fn _wrap_tket_conversion<F: FnOnce(Circuit) -> Circuit>(
    f: F,
) -> impl FnOnce(Py<PyAny>) -> PyResult<Py<PyAny>> {
    |c: Py<PyAny>| (f)(Circuit::_from_tket1(c)).to_tket1()
}

#[pyfunction]
fn remove_redundancies(c: Py<PyAny>) -> PyResult<Py<PyAny>> {
    _wrap_tket_conversion(|circ| tket_rs::passes::squash::cx_cancel_pass(circ).0)(c)
}

#[pyfunction]
fn greedy_rewrite(circ: Circuit, pattern: Circuit, rewrite_fn: Py<PyAny>) -> Circuit {
    assert!(Python::with_gil(|py| rewrite_fn.as_ref(py).is_callable()));

    apply_greedy(circ, |c| {
        // todo allow spec of node comp closure too
        let pattern = CircFixedStructPattern::from_circ(pattern.clone(), node_equality());
        pattern_rewriter(pattern, c, |m| {
            let outc: Circuit = Python::with_gil(|py| {
                let pd = m.into_py(py);
                rewrite_fn.call1(py, (pd,)).unwrap().extract(py).unwrap()
            });
            (outc, 0.0)
        })
        .next()
    })
    .unwrap()
    .0
}
/// A Python module implemented in Rust.
#[pymodule]
fn pyrs(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(remove_redundancies, m)?)?;
    m.add_function(wrap_pyfunction!(greedy_rewrite, m)?)?;
    m.add_class::<Circuit>()?;
    m.add_class::<PyOp>()?;
    m.add_class::<WireType>()?;
    m.add_class::<PyOpenCircuit>()?;
    m.add_class::<PySubgraph>()?;
    m.add_class::<CircuitRewrite>()?;
    Ok(())
}
