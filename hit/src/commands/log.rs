//! Log command implementation.

use anyhow::{anyhow, Context, Result};
use hugr::HugrView;

use super::CommandExecutor;
use crate::config::Config;
use crate::storage::RewriteSpaceData;

#[derive(Debug)]
pub struct LogCommand {
    pub all: bool,
}

impl CommandExecutor for LogCommand {
    fn execute(&self) -> Result<()> {
        let config = Config::load_or_default()?;

        let current_file = config
            .current_file
            .as_ref()
            .ok_or_else(|| anyhow!("No current file loaded. Use 'hit load' first."))?;

        // Load the rewrite space data
        let data = RewriteSpaceData::load(current_file)
            .with_context(|| format!("Failed to load rewrite space from {}", current_file))?;

        let (mut commits_to_show, title) = if self.all {
            // Show all commits in the space
            let all_commit_ids: Vec<_> = data.space.all_commit_ids().collect();
            (all_commit_ids, "All commits in the space:")
        } else {
            // Show only selected commits and their ancestors (original behavior)
            let selected_commit_ids = data.current_hugr().all_commit_ids().collect();
            (selected_commit_ids, "Selected commits and their ancestors:")
        };

        if commits_to_show.is_empty() {
            if self.all {
                println!("No commits found in the space.");
            } else {
                println!("No commits currently selected.");
            }
            return Ok(());
        }

        // Sort commits chronologically by timestamp
        commits_to_show
            .sort_by_key(|&commit_id| data.space.get_timestamp(commit_id).unwrap_or_default());

        println!("{}", title);
        println!("{}", "=".repeat(title.len()));

        // Print information about each commit
        for &commit_id in &commits_to_show {
            let commit = data.space.get_commit(commit_id);
            match commit.replacement() {
                Some(replacement) => {
                    let added_nodes = commit.inserted_nodes().count();
                    let removed_nodes = replacement.subgraph().nodes().len();
                    println!(
                        "  {} - +{} nodes, -{} nodes",
                        commit_id, added_nodes, removed_nodes
                    );
                }
                None => {
                    println!("  {} - base commit", commit_id);
                }
            }
        }

        println!();
        if self.all {
            println!("Total commits: {}", commits_to_show.len());
        } else {
            // Show the final extracted hugr info for selected commits
            let current_hugr = data.current_hugr();
            let extracted_hugr = current_hugr.to_hugr();
            println!(
                "Extracted HUGR has {} nodes total",
                extracted_hugr.num_nodes()
            );
            println!("Selected {} commits", commits_to_show.len());
        }

        Ok(())
    }
}
