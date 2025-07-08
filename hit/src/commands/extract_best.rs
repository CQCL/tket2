//! Extract best command implementation.

use anyhow::{anyhow, Context, Result};
use hugr::HugrView;

use super::CommandExecutor;
use crate::config::Config;
use crate::display::CommitHexId;
use crate::storage::LoadedRewriteSpace;

#[derive(Debug)]
pub struct ExtractBestCommand;

impl CommandExecutor for ExtractBestCommand {
    fn execute(&self) -> Result<()> {
        // Load the rewrite space data
        let mut config = Config::load_or_default()?;
        let data = LoadedRewriteSpace::load_from_config(&config)?;

        // Extract the best rewrite sequence
        let best_hugr = data
            .space
            .extract_best_with_cost(|c| c.rewrite_cost.clone())
            .ok_or_else(|| anyhow!("Failed to find optimal rewrite sequence"))?;

        // Get the commits that were selected for the best solution
        let selected_commits: Vec<_> = best_hugr.all_commit_ids().collect();
        let commit_strings: Vec<String> = selected_commits
            .iter()
            .map(|&c| format!("{}", CommitHexId(c)))
            .collect();

        println!(
            "Found optimal rewrite sequence with {} commits: {}",
            selected_commits.len(),
            commit_strings.join(", ")
        );

        // Update the selected commits in our data
        let selected_commit_ids = data
            .try_select_commits(&commit_strings)
            .context("Failed to set selected commits")?;

        config.set_selected_commits(&selected_commit_ids)?;

        let final_hugr = best_hugr.to_hugr();
        println!();
        println!(
            "Selected commits updated. Final HUGR has {} nodes.",
            final_hugr.num_nodes()
        );

        Ok(())
    }
}
