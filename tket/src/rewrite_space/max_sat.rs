use std::{collections::BTreeMap, fmt::Debug};

use itertools::Itertools;
use z3::ast::Bool as Z3Bool;

type Symbol = String;

/// A solver for a particular Max-SAT problem.
///
/// It will ensure that all constraints:
///  - a => b for all (a, b) in `self.implications`.
///  - ¬(a & b) for all (a, b) in `self.mutexes`.
///
/// are satisfied, whilst setting as many variables as possible to true.
pub(super) struct MaxSATSolver {
    implications: Vec<(Symbol, Symbol)>,
    mutex: Vec<(Symbol, Symbol)>,
    weights: BTreeMap<Symbol, isize>,
}

impl Debug for MaxSATSolver {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self.to_z3())
    }
}

impl MaxSATSolver {
    pub(super) fn new() -> Self {
        Self::default()
    }

    pub(super) fn get_model(&self) -> Option<BTreeMap<Symbol, bool>> {
        let z3_opt = self.to_z3();
        let z3_model = match z3_opt.check(&[]) {
            z3::SatResult::Sat => Some(z3_opt.get_model().expect("check just succeeded")),
            _ => None,
        }?;

        Some(
            self.all_symbols()
                .map(|symb| {
                    (
                        symb.to_string(),
                        z3_model
                            .get_const_interp(&self.z3_symbol(symb))
                            .expect("interpretation for symbol")
                            .as_bool()
                            .expect("concrete interpretation"),
                    )
                })
                .collect(),
        )
    }

    pub(super) fn extend_implications<S: Into<Symbol>>(
        &mut self,
        implications: impl IntoIterator<Item = (S, S)>,
    ) {
        self.implications
            .extend(implications.into_iter().map(|(a, b)| (a.into(), b.into())));
    }

    #[allow(unused)]
    pub(super) fn extend_mutexes<S: Into<Symbol>>(
        &mut self,
        mutexes: impl IntoIterator<Item = (S, S)>,
    ) {
        self.mutex
            .extend(mutexes.into_iter().map(|(a, b)| (a.into(), b.into())));
    }

    #[allow(unused)]
    pub(super) fn add_implication(&mut self, a: impl Into<Symbol>, b: impl Into<Symbol>) {
        self.implications.push((a.into(), b.into()));
    }

    pub(super) fn add_mutex(&mut self, a: impl Into<Symbol>, b: impl Into<Symbol>) {
        self.mutex.push((a.into(), b.into()));
    }

    pub(super) fn set_weight(&mut self, symb: impl Into<Symbol>, weight: isize) {
        self.weights.insert(symb.into(), weight);
    }

    fn z3_implications(&self) -> impl Iterator<Item = Z3Bool> + '_ {
        self.implications
            .iter()
            .map(|(a, b)| Z3Bool::implies(&self.z3_symbol(a), &self.z3_symbol(b)))
    }

    fn z3_mutexes(&self) -> impl Iterator<Item = Z3Bool> + '_ {
        self.mutex
            .iter()
            .map(|(a, b)| !(self.z3_symbol(a) & self.z3_symbol(b)))
    }

    pub(super) fn z3_symbol(&self, symb: &str) -> Z3Bool {
        Z3Bool::new_const(symb)
    }

    pub(super) fn all_symbols(&self) -> impl Iterator<Item = &Symbol> {
        self.implications
            .iter()
            .chain(self.mutex.iter())
            .flat_map(|(a, b)| [a, b])
            .chain(self.weights.keys())
            .unique()
    }

    fn to_z3(&self) -> z3::Optimize {
        // Create an optimizer
        let opt = z3::Optimize::new();

        // Add the hard constraints
        for a_implies_b in self.z3_implications() {
            opt.assert(&a_implies_b);
        }

        // Add the mutual exclusivity constraints
        for not_a_and_b in self.z3_mutexes() {
            opt.assert(&not_a_and_b);
        }

        // Add the soft constraints (optimisation objective)
        for symb in self.all_symbols() {
            let weight = self.weights.get(symb).copied().unwrap_or(1);
            opt.assert_soft(&self.z3_symbol(symb), weight, None);
        }

        opt
    }
}

impl Default for MaxSATSolver {
    fn default() -> Self {
        Self {
            implications: vec![],
            mutex: vec![],
            weights: BTreeMap::new(),
        }
    }
}
