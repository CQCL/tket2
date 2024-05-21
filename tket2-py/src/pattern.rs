//! Pattern matching on circuits.

pub mod portmatching;

use crate::circuit::Tk2Circuit;
use crate::rewrite::PyCircuitRewrite;
use crate::utils::{create_py_exception, ConvertPyErr};

use hugr::Hugr;
use pyo3::prelude::*;
use tket2::portmatching::{CircuitPattern, PatternMatcher};

/// The module definition
pub fn module(py: Python<'_>) -> PyResult<Bound<'_, PyModule>> {
    let m = PyModule::new_bound(py, "pattern")?;
    m.add_class::<Rule>()?;
    m.add_class::<RuleMatcher>()?;
    m.add_class::<self::portmatching::PyCircuitPattern>()?;
    m.add_class::<self::portmatching::PyPatternMatcher>()?;
    m.add_class::<self::portmatching::PyPatternMatch>()?;
    m.add_class::<self::portmatching::PyPatternID>()?;

    m.add(
        "InvalidPatternError",
        py.get_type_bound::<PyInvalidPatternError>(),
    )?;
    m.add(
        "InvalidReplacementError",
        py.get_type_bound::<PyInvalidReplacementError>(),
    )?;

    Ok(m)
}

create_py_exception!(
    hugr::hugr::views::sibling_subgraph::InvalidReplacement,
    PyInvalidReplacementError,
    "Errors that can occur while constructing a HUGR replacement."
);

create_py_exception!(
    tket2::portmatching::pattern::InvalidPattern,
    PyInvalidPatternError,
    "Conversion error from circuit to pattern."
);

#[derive(Clone)]
#[pyclass]
/// A rewrite rule defined by a left hand side and right hand side of an equation.
pub struct Rule(pub [Hugr; 2]);

#[pymethods]
impl Rule {
    #[new]
    fn new_rule(l: &Bound<PyAny>, r: &Bound<PyAny>) -> PyResult<Rule> {
        let l = Tk2Circuit::new(l)?;
        let r = Tk2Circuit::new(r)?;

        Ok(Rule([l.hugr, r.hugr]))
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
        let matcher = PatternMatcher::from_patterns(patterns.convert_pyerrs()?);

        Ok(Self { matcher, rights })
    }

    pub fn find_match(&self, target: &Tk2Circuit) -> PyResult<Option<PyCircuitRewrite>> {
        let h = &target.hugr;
        if let Some(p_match) = self.matcher.find_matches_iter(h).next() {
            let r = self.rights.get(p_match.pattern_id().0).unwrap().clone();
            let rw = p_match.to_rewrite(h, r).convert_pyerrs()?;
            Ok(Some(rw.into()))
        } else {
            Ok(None)
        }
    }
}
