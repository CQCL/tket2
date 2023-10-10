use std::{cmp::min, convert::TryInto, fs, num::NonZeroUsize, path::PathBuf};

use pyo3::{prelude::*, types::IntoPyDict};
use tket2::{json::TKETDecode, op_matches, passes::apply_greedy_commutation, Circuit, T2Op};
use tket_json_rs::circuit_json::SerialCircuit;

use crate::{
    circuit::{try_update_hugr, try_with_hugr},
    optimiser::PyDefaultTasoOptimiser,
};

#[pyfunction]
fn greedy_depth_reduce(py_c: PyObject) -> PyResult<(PyObject, u32)> {
    try_with_hugr(py_c, |mut h| {
        let n_moves = apply_greedy_commutation(&mut h)?;
        let py_c = SerialCircuit::encode(&h)?.to_tket1()?;
        PyResult::Ok((py_c, n_moves))
    })
}

/// TASO optimisation pass.
///
/// HyperTKET's best attempt at optimising a circuit using circuit rewriting
/// and TASO.
///
/// Input can be in any gate set and will be rebased to Nam, i.e. CX + Rz + H.
///
/// Will use at most `max_threads` threads (plus a constant) and take at most
/// `timeout` seconds (plus a constant). Default to the number of cpus and
/// 15min respectively.
///
/// Log files will be written to the directory `log_dir` if specified.
///
/// This requires a `nam_6_3.rwr` file in the current directory. The location
/// can alternatively be specified using the `rewriter_dir` argument.
#[pyfunction]
fn taso_optimise(
    circ: PyObject,
    max_threads: Option<NonZeroUsize>,
    rewriter_dir: Option<PathBuf>,
    timeout: Option<u64>,
    log_dir: Option<PathBuf>,
) -> PyResult<PyObject> {
    // Runs the following code:
    // ```python
    // from pytket.passes.auto_rebase import auto_rebase_pass
    // from pytket import OpType
    // auto_rebase_pass({OpType.CX, OpType.Rz, OpType.H}).apply(circ)"
    // ```
    Python::with_gil(|py| {
        let auto_rebase = py
            .import("pytket.passes.auto_rebase")?
            .getattr("auto_rebase_pass")?;
        let optype = py.import("pytket")?.getattr("OpType")?;
        let locals = [("OpType", &optype)].into_py_dict(py);
        let op_set = py.eval("{OpType.CX, OpType.Rz, OpType.H}", None, Some(locals))?;
        let rebase_pass = auto_rebase.call1((op_set,))?.getattr("apply")?;
        rebase_pass.call1((&circ,)).map(|_| ())
    })?;
    nam_taso_optimise(circ, max_threads, rewriter_dir, timeout, log_dir)
}

/// TASO optimisation pass.
///
/// HyperTKET's best attempt at optimising a circuit using circuit rewriting
/// and TASO.
///
/// Input must be in the Nam gate set, i.e. CX + Rz + H.
///
/// Will use at most `max_threads` threads (plus a constant) and take at most
/// `timeout` seconds (plus a constant). Default to the number of cpus and
/// 30s respectively.
///
/// Log files will be written to the directory `log_dir` if specified.
///
/// This requires a `nam_6_3.rwr` file in the current directory. The location
/// can alternatively be specified using the `rewriter_dir` argument.
#[pyfunction]
fn nam_taso_optimise(
    circ: PyObject,
    max_threads: Option<NonZeroUsize>,
    rewriter_dir: Option<PathBuf>,
    timeout: Option<u64>,
    log_dir: Option<PathBuf>,
) -> PyResult<PyObject> {
    let max_threads = max_threads.unwrap_or(num_cpus::get().try_into().unwrap());
    let rewrite_dir = rewriter_dir.unwrap_or(PathBuf::from("."));
    let timeout = timeout.unwrap_or(30);
    if let Some(log_dir) = log_dir.as_ref() {
        fs::create_dir_all(log_dir)?;
    }
    let taso_splits = |n_threads: NonZeroUsize| match n_threads.get() {
        n if n >= 7 => (
            vec![n, 3, 1],
            vec![timeout / 2, timeout / 10 * 3, timeout / 10 * 2],
        ),
        n if n >= 4 => (
            vec![n, 2, 1],
            vec![timeout / 2, timeout / 10 * 3, timeout / 10 * 2],
        ),
        n if n > 1 => (vec![n, 1], vec![timeout / 2, timeout / 2]),
        1 => (vec![1], vec![timeout]),
        _ => unreachable!(),
    };
    let optimiser = PyDefaultTasoOptimiser::load_precompiled(rewrite_dir.join("nam_6_3.rwr"));
    try_update_hugr(circ, |mut circ| {
        let n_cx = circ
            .commands()
            .filter(|c| op_matches(c.optype(), T2Op::CX))
            .count();
        let n_threads = min(
            (n_cx / 50).try_into().unwrap_or(1.try_into().unwrap()),
            max_threads,
        );
        let (split_threads, split_timeouts) = taso_splits(n_threads);
        for (i, (n_threads, timeout)) in split_threads.into_iter().zip(split_timeouts).enumerate() {
            let log_file = log_dir.as_ref().map(|log_dir| {
                let mut log_file = log_dir.clone();
                log_file.push(format!("cycle-{i}.log"));
                log_file
            });
            circ = optimiser.optimise(
                circ,
                Some(timeout),
                Some(n_threads.try_into().unwrap()),
                Some(true),
                log_file,
            );
        }
        PyResult::Ok(circ)
    })
}

pub(crate) fn add_pass_module(py: Python, parent: &PyModule) -> PyResult<()> {
    let m = PyModule::new(py, "passes")?;
    m.add_function(wrap_pyfunction!(greedy_depth_reduce, m)?)?;
    m.add_function(wrap_pyfunction!(nam_taso_optimise, m)?)?;
    m.add_function(wrap_pyfunction!(taso_optimise, m)?)?;
    m.add_class::<tket2::T2Op>()?;
    m.add(
        "PullForwardError",
        py.get_type::<tket2::passes::PyPullForwardError>(),
    )?;
    parent.add_submodule(m)?;
    Ok(())
}
