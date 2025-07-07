//! Display utilities for commit information.

use hugr::persistent::CommitId;
use tket2::rewrite_space::RewriteSpace;

/// Display information about a single commit
pub fn display_commit(space: &RewriteSpace<isize>, commit_id: CommitId) -> String {
    let commit = space.get_commit(commit_id);
    let name = space.get_name(commit_id).unwrap_or("unnamed");
    match commit.replacement() {
        Some(replacement) => {
            let added_nodes = commit.inserted_nodes().count();
            let removed_nodes = replacement.subgraph().nodes().len();
            format!(
                "  {} - {}. (+{}, -{})",
                commit_id, name, added_nodes, removed_nodes
            )
        }
        None => format!("  {} - base commit", commit_id,),
    }
}

/// Display a list of commits with a title
pub fn display_commits<'a>(
    space: &RewriteSpace<isize>,
    commits: impl IntoIterator<Item = &'a CommitId>,
    title: &str,
) {
    if title.len() > 0 {
        println!("{}", title);
        println!("{}", "=".repeat(title.len()));
    }

    for &commit_id in commits {
        println!("{}", display_commit(space, commit_id));
    }
}
