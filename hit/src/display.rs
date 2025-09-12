//! Display utilities for commit information.

use std::hash::{Hash, Hasher};

use derive_more::derive::{From, Into};
use fxhash::FxHasher64;
use hugr::persistent::{Commit, CommitId};

/// A commit ID formatted as a hexadecimal string.
#[derive(Debug, Clone, From, Into)]
pub struct CommitHexId(pub CommitId);
impl std::fmt::Display for CommitHexId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let dbg_fmt = format!("{:?}", self.0);
        let mut hasher = FxHasher64::default();
        dbg_fmt.hash(&mut hasher);
        write!(f, "{:x}", hasher.finish())
    }
}

/// Display information about a single commit
pub fn display_commit(commit: &Commit) -> String {
    // let name = space.get_name(commit.id()).unwrap_or("unnamed");
    let name = "unnamed";
    match commit.replacement() {
        Some(replacement) => {
            let added_nodes = commit.inserted_nodes().count();
            let removed_nodes = replacement.subgraph().nodes().len();
            format!(
                "  {} - {}. (+{}, -{})",
                CommitHexId(commit.id()),
                name,
                added_nodes,
                removed_nodes
            )
        }
        None => format!("  {} - base commit", CommitHexId(commit.id())),
    }
}

/// Display a list of commits with a title
pub fn display_commits<'a: 'b, 'b>(commits: impl IntoIterator<Item = &'b Commit<'a>>, title: &str) {
    if title.len() > 0 {
        println!("{}", title);
        println!("{}", "=".repeat(title.len()));
    }

    for commit in commits {
        println!("{}", display_commit(commit));
    }
}
