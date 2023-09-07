//! A rewriter based on circuit equivalence classes.
//!
//! Circuits are clustered in equivalence classes based on whether they
//! represent the same unitary.
//!
//! This rewriter uses the [`CircuitMatcher`] to find known subcircuits and
//! generates rewrites to replace them with other circuits within the same
//! equivalence class.
//!
//! Equivalence classes are generated using Quartz.

use derive_more::{From, Into};
use itertools::Itertools;
use portmatching::PatternID;
use std::path::Path;

use hugr::{
    hugr::views::{HierarchyView, SiblingGraph},
    ops::handle::DfgID,
    Hugr, HugrView,
};

use crate::{
    circuit::Circuit,
    passes::taso::{load_eccs_json_file, EqCircClass},
    portmatching::{CircuitMatcher, CircuitPattern},
};

use super::{CircuitRewrite, Rewriter};

#[derive(Debug, Clone, Copy, PartialEq, Eq, From, Into)]
struct TargetID(usize);

/// A rewriter based on circuit equivalence classes.
///
/// In every equivalence class, one circuit is chosen as the representative.
/// Valid rewrites turn a non-representative circuit into its representative,
/// or a representative circuit into any of the equivalent non-representative
/// circuits.
pub struct ECCRewriter {
    /// Matcher for finding patterns.
    matcher: CircuitMatcher,
    /// Targets of some rewrite rules.
    targets: Vec<Hugr>,
    /// Rewrites, stored as a map from PatternID to TargetIDs.
    rewrite_rules: Vec<Vec<TargetID>>,
}

impl ECCRewriter {
    /// Create a new rewriter from equivalent circuit classes in JSON file.
    ///
    /// This uses the Quartz JSON file format to store equivalent circuit classes.
    pub fn from_eccs_json_file(path: impl AsRef<Path>) -> Self {
        let eccs = load_eccs_json_file(path);
        Self::from_eccs(eccs)
    }

    /// Create a new rewriter from a list of equivalent circuit classes.
    ///
    /// Equivalence classes are represented as [`EqCircClass`]s, lists of
    /// HUGRs where one of the elements is chosen as the representative.
    pub fn from_eccs(eccs: impl Into<Vec<EqCircClass>>) -> Self {
        let eccs = eccs.into();
        let rewrite_rules = get_rewrite_rules(&eccs);
        let patterns = get_patterns(&eccs);
        // Remove failed patterns
        let (patterns, rewrite_rules): (Vec<_>, Vec<_>) = patterns
            .into_iter()
            .zip(rewrite_rules)
            .filter_map(|(p, r)| Some((p?, r)))
            .unzip();
        let targets = into_targets(eccs);
        let matcher = CircuitMatcher::from_patterns(patterns);
        Self {
            matcher,
            targets,
            rewrite_rules,
        }
    }

    /// Get all targets of rewrite rules given a source pattern.
    fn get_targets(&self, pattern: PatternID) -> impl Iterator<Item = &Hugr> {
        self.rewrite_rules[pattern.0]
            .iter()
            .map(|id| &self.targets[id.0])
    }
}

impl Rewriter for ECCRewriter {
    fn get_rewrites<'a, C: Circuit<'a>>(&'a self, circ: &'a C) -> Vec<CircuitRewrite> {
        let matches = self.matcher.find_matches(circ);
        matches
            .into_iter()
            .flat_map(|m| {
                let pattern_id = m.pattern_id();
                self.get_targets(pattern_id)
                    .map(move |repl| m.to_rewrite(repl.clone()).expect("invalid replacement"))
            })
            .collect()
    }
}

fn into_targets(rep_sets: Vec<EqCircClass>) -> Vec<Hugr> {
    rep_sets
        .into_iter()
        .flat_map(|rs| rs.into_circuits())
        .collect()
}

fn get_rewrite_rules(rep_sets: &[EqCircClass]) -> Vec<Vec<TargetID>> {
    let n_circs = rep_sets.iter().map(|rs| rs.len()).sum::<usize>();
    let mut rewrite_rules = vec![Default::default(); n_circs];
    let mut curr_target = 0;
    for rep_set in rep_sets {
        let rep_ind = curr_target;
        let other_inds = (curr_target + 1)..(curr_target + rep_set.len());
        // Rewrite rules for representative circuit
        rewrite_rules[rep_ind] = other_inds.clone().map_into().collect();
        // Rewrite rules for other circuits
        for i in other_inds {
            rewrite_rules[i] = vec![rep_ind.into()];
        }
        curr_target += rep_set.len();
    }
    rewrite_rules
}

fn get_patterns(rep_sets: &[EqCircClass]) -> Vec<Option<CircuitPattern>> {
    let all_hugrs = rep_sets.iter().flat_map(|rs| rs.circuits());
    let all_circs = all_hugrs
        .map(|hugr| SiblingGraph::<DfgID>::new(hugr, hugr.root()))
        // TODO: collect to vec because of lifetime issues that should not exist
        .collect_vec();
    all_circs
        .iter()
        .map(|circ| CircuitPattern::try_from_circuit(circ).ok())
        .collect()
}

#[cfg(test)]
mod tests {
    use crate::{utils::build_simple_circuit, T2Op};

    use super::*;

    fn empty() -> Hugr {
        build_simple_circuit(2, |_| Ok(())).unwrap()
    }

    fn h_h() -> Hugr {
        build_simple_circuit(2, |circ| {
            circ.append(T2Op::H, [0]).unwrap();
            circ.append(T2Op::H, [0]).unwrap();
            Ok(())
        })
        .unwrap()
    }

    fn cx_cx() -> Hugr {
        build_simple_circuit(2, |circ| {
            circ.append(T2Op::CX, [0, 1]).unwrap();
            circ.append(T2Op::CX, [0, 1]).unwrap();
            Ok(())
        })
        .unwrap()
    }

    fn cx_x() -> Hugr {
        build_simple_circuit(2, |circ| {
            circ.append(T2Op::CX, [0, 1]).unwrap();
            circ.append(T2Op::X, [1]).unwrap();
            Ok(())
        })
        .unwrap()
    }

    fn x_cx() -> Hugr {
        build_simple_circuit(2, |circ| {
            circ.append(T2Op::X, [1]).unwrap();
            circ.append(T2Op::CX, [0, 1]).unwrap();
            Ok(())
        })
        .unwrap()
    }

    #[test]
    fn small_ecc_rewriter() {
        let ecc1 = EqCircClass::new(h_h(), vec![empty(), cx_cx()]);
        let ecc2 = EqCircClass::new(cx_x(), vec![x_cx()]);
        let rewriter = ECCRewriter::from_eccs(vec![ecc1, ecc2]);
        assert_eq!(rewriter.targets.len(), 5);
        assert_eq!(
            rewriter.rewrite_rules,
            [
                vec![TargetID(1), TargetID(2)],
                vec![TargetID(0)],
                vec![TargetID(4)],
                vec![TargetID(3)],
            ]
        );
        assert_eq!(rewriter.get_targets(PatternID(1)).collect_vec(), [&h_h()]);
    }

    #[test]
    fn ecc_rewriter_from_file() {
        // In this example, all circuits are valid patterns, thus
        // PatternID == TargetID.
        let test_file = "test_files/small_eccs.json";
        let rewriter = ECCRewriter::from_eccs_json_file(test_file);
        assert_eq!(rewriter.rewrite_rules.len(), rewriter.matcher.n_patterns());
        assert_eq!(rewriter.targets.len(), 5 * 4 + 4 * 3);

        // Assert that the rewrite rules are correct, i.e that the rewrite
        // rules in the slice (k..=k+t) is given by [[k+1, ..., k+t], [k], ..., [k]]
        // where k is the index of a representative circuit and t+1 is the size
        // of the ECC.
        let mut n_eccs_of_len = [0; 4];
        let mut next_k_are_1 = 0;
        let mut curr_repr = TargetID(0);
        for (i, rws) in rewriter.rewrite_rules.into_iter().enumerate() {
            n_eccs_of_len[rws.len()] += 1;
            if rws.len() == 1 {
                assert!(next_k_are_1 > 0);
                assert_eq!(rws, vec![curr_repr]);
                next_k_are_1 -= 1;
            } else {
                assert_eq!(next_k_are_1, 0);
                let exp_rws: Vec<_> = (i + 1..=i + rws.len()).map(TargetID).collect();
                assert_eq!(rws, exp_rws);
                next_k_are_1 = rws.len();
                curr_repr = TargetID(i);
            }
        }
        // There should be 4x ECCs of size 3 and 5x ECCs of size 4
        let exp_n_eccs_of_len = [0, 4 * 2 + 5 * 3, 4, 5];
        assert_eq!(n_eccs_of_len, exp_n_eccs_of_len);
    }
}
