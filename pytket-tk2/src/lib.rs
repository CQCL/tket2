//! Python bindings for TKET2.
#![warn(missing_docs)]

mod circuit;
mod optimiser;
mod pass;

use circuit::{add_circuit_module, to_hugr, T2Circuit};
use optimiser::add_optimiser_module;
use pass::add_pass_module;

use hugr::Hugr;
use pyo3::prelude::*;
use tket2::{
    portmatching::{pyo3::PyPatternMatch, CircuitPattern, PatternMatcher},
    rewrite::CircuitRewrite,
};

#[derive(Clone)]
#[pyclass]
/// A rewrite rule defined by a left hand side and right hand side of an equation.
pub struct Rule(pub [Hugr; 2]);

#[pymethods]
impl Rule {
    #[new]
    fn new_rule(l: PyObject, r: PyObject) -> PyResult<Rule> {
        let l = to_hugr(l)?;
        let r = to_hugr(r)?;

        Ok(Rule([l, r]))
    }
}
#[pyclass]
struct RuleMatcher {
    matcher: PatternMatcher,
    rights: Vec<Hugr>,
}

#[pymethods]
impl RuleMatcher {
    #[new]
    pub fn from_rules(rules: Vec<Rule>) -> PyResult<Self> {
        let (lefts, rights): (Vec<_>, Vec<_>) =
            rules.into_iter().map(|Rule([l, r])| (l, r)).unzip();
        let patterns: Result<Vec<CircuitPattern>, _> =
            lefts.iter().map(CircuitPattern::try_from_circuit).collect();
        let matcher = PatternMatcher::from_patterns(patterns?);

        Ok(Self { matcher, rights })
    }

    pub fn find_match(&self, target: &T2Circuit) -> PyResult<Option<CircuitRewrite>> {
        let h = &target.0;
        let p_match = self.matcher.find_matches_iter(h).next();
        if let Some(m) = p_match {
            let py_match = PyPatternMatch::try_from_rust(m, h, &self.matcher)?;
            let r = self.rights.get(py_match.pattern_id).unwrap().clone();
            let rw = py_match.to_rewrite(h, r)?;
            Ok(Some(rw))
        } else {
            Ok(None)
        }
    }
}

/// The Python bindings to TKET2.
#[pymodule]
fn pytket_tk2(py: Python, m: &PyModule) -> PyResult<()> {
    add_circuit_module(py, m)?;
    add_pattern_module(py, m)?;
    add_pass_module(py, m)?;
    add_optimiser_module(py, m)?;
    Ok(())
}

/// portmatching module
fn add_pattern_module(py: Python, parent: &PyModule) -> PyResult<()> {
    let m = PyModule::new(py, "pattern")?;
    m.add_class::<tket2::portmatching::CircuitPattern>()?;
    m.add_class::<tket2::portmatching::PatternMatcher>()?;
    m.add_class::<CircuitRewrite>()?;
    m.add_class::<Rule>()?;
    m.add_class::<RuleMatcher>()?;

    m.add(
        "InvalidPatternError",
        py.get_type::<tket2::portmatching::pattern::PyInvalidPatternError>(),
    )?;
    m.add(
        "InvalidReplacementError",
        py.get_type::<hugr::hugr::views::sibling_subgraph::PyInvalidReplacementError>(),
    )?;

    parent.add_submodule(m)
}
