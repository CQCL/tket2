use crate::rewrite_space::Walker;

use super::{CommitFactory, IterMatched};
use delegate::delegate;
use hugr::persistent::{Commit, PatchNode};

/// A dyn-compatible trait that captures the essence of a commit factory:
/// producing a sequence of commits.
///
/// Taking a pattern root and a walker as arguments, an implementation of
/// [`FindCommits`] should return the commits that correspond to rewrites of
/// subgraphs discovered from the root using the walker. These are typically
/// obtained from commit factories by finding pattern matching and creating
/// rewrites replacing matches with their replacement.
///
/// An implementation is provided for any `T: CommitFactory` as well as
/// vectors of boxed commit factories [`BoxedCommitFactory`].
///
/// ```
/// use hugr::{Hugr, HugrView};
/// use tket2::rewrite_space::{BoxedCommitFactory, FindCommits, PersistentHugr, Walker};
/// use tket2::rewrite_space::{CommuteCZFactory};
///
/// let factories: Vec<BoxedCommitFactory> =
///     vec![CommuteCZFactory.into(), CommuteCZFactory.into()];
///
/// let phugr = PersistentHugr::with_base(Hugr::new());
/// let pattern_root = phugr.module_root();
/// let walker = Walker::from_pinned_node(pattern_root, phugr.as_state_space());
///
/// let commits = factories.find_commits(pattern_root, walker);
/// assert_eq!(commits.len(), 0);
/// ```
pub trait FindCommits {
    /// Find commits starting from the given pattern root using the walker.
    fn find_commits(&self, pattern_root: PatchNode, walker: Walker) -> Vec<Commit>;
}

impl<T: CommitFactory> FindCommits for T {
    fn find_commits(&self, pattern_root: PatchNode, walker: Walker) -> Vec<Commit> {
        self.find_pattern_matches(pattern_root, walker)
            .into_iter()
            .filter_map(|(pattern_match, new_walker)| {
                let host = new_walker.as_hugr_view();
                let subgraph = pattern_match.to_subgraph(&new_walker).expect("valid match");
                let Ok(sibling_subgraph) = subgraph.to_sibling_subgraph(host) else {
                    // not convex
                    return None;
                };
                let Some(replacement) =
                    self.get_replacement(&pattern_match, &sibling_subgraph, host)
                else {
                    return None;
                };
                new_walker
                    .try_create_commit(subgraph, replacement, |node, port| {
                        self.map_boundary(node, port, &pattern_match, host)
                    })
                    .ok()
            })
            .collect()
    }
}

impl FindCommits for Vec<BoxedCommitFactory> {
    fn find_commits(&self, pattern_root: PatchNode, walker: Walker) -> Vec<Commit> {
        self.iter()
            .flat_map(|factory| factory.find_commits(pattern_root, walker.clone()))
            .collect()
    }
}

/// An arbitrary commit factory boxed into a trait object.
pub struct BoxedCommitFactory(Box<dyn FindCommits>);

impl BoxedCommitFactory {
    /// Create a new boxed commit factory from a commit factory.
    pub fn new(value: impl FindCommits + 'static) -> Self {
        Self(Box::new(value))
    }
}

impl<T: CommitFactory + 'static> From<T> for BoxedCommitFactory {
    fn from(value: T) -> Self {
        BoxedCommitFactory::new(value)
    }
}

impl FindCommits for BoxedCommitFactory {
    delegate! {
        to self.0 {
            fn find_commits(&self, pattern_root: PatchNode, walker: Walker) -> Vec<Commit>;
        }
    }
}
