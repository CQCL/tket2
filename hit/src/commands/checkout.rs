//! Checkout command implementation.

use anyhow::{anyhow, Context, Result};
use hugr::HugrView;

use super::CommandExecutor;
use crate::config::Config;
use crate::storage::RewriteSpaceData;

#[derive(Debug)]
pub struct CheckoutCommand {
    pub commits: Vec<String>,
}

impl CommandExecutor for CheckoutCommand {
    fn execute(&self) -> Result<()> {
        let mut config = Config::load_or_default()?;

        let current_file = config
            .current_file
            .as_ref()
            .ok_or_else(|| anyhow!("No current file loaded. Use 'hit load' first."))?;

        // Load the rewrite space data
        let mut data = RewriteSpaceData::load(current_file)
            .with_context(|| format!("Failed to load rewrite space from {}", current_file))?;

        // Use the centralized method to set selected commits
        data.set_selected_commits(&self.commits)?;

        // Get the selected commit IDs for display and config storage
        let selected_commit_ids = data.get_selected_commit_ids();

        // Update config with the selected commits
        config.selected_commits = selected_commit_ids
            .iter()
            .map(|c| format!("{}", c))
            .collect();
        config.save()?;

        // Save the updated data
        data.save()?;

        println!(
            "Successfully selected {} commits:",
            selected_commit_ids.len()
        );
        for commit_id in &selected_commit_ids {
            let commit = data.space.get_commit(*commit_id);
            match commit.replacement() {
                Some(replacement) => {
                    let added_nodes = commit.inserted_nodes().count();
                    let removed_nodes = replacement.subgraph().nodes().len();
                    println!(
                        "  {:?} - +{} nodes, -{} nodes",
                        commit_id, added_nodes, removed_nodes
                    );
                }
                None => {
                    println!("  {:?} - base commit", commit_id);
                }
            }
        }

        Ok(())
    }
}
