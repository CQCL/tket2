//! Log command implementation.

use anyhow::Result;
use hugr::HugrView;

use super::CommandExecutor;
use crate::config::Config;
use crate::display::display_commits;
use crate::storage::RewriteSpaceData;

#[derive(Debug)]
pub struct LogCommand {
    pub all: bool,
}

impl CommandExecutor for LogCommand {
    fn execute(&self) -> Result<()> {
        // Load the rewrite space data
        let config = Config::load_or_default()?;
        let data = RewriteSpaceData::load_from_config(&config)?;

        let (mut commits_to_show, title) = if self.all {
            // Show all commits in the space
            let all_commit_ids: Vec<_> = data.space.all_commit_ids().collect();
            (all_commit_ids, "All commits in the space:")
        } else {
            // Show only selected commits and their ancestors (original behavior)
            let selected_commit_ids = data.current_hugr(&config)?.all_commit_ids().collect();
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

        commits_to_show.reverse();

        display_commits(&data.space, &commits_to_show, &title);

        println!();
        if self.all {
            println!("Total commits: {}", commits_to_show.len());
        } else {
            // Show the final extracted hugr info for selected commits
            let current_hugr = data.current_hugr(&config)?;
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
