//! Checkout command implementation.

use anyhow::Result;

use super::CommandExecutor;
use crate::config::Config;
use crate::display::display_commits;
use crate::storage::LoadedRewriteSpace;

#[derive(Debug)]
pub struct CheckoutCommand {
    pub commits: Vec<String>,
}

impl CommandExecutor for CheckoutCommand {
    fn execute(&self) -> Result<()> {
        let mut config = Config::load_or_default()?;

        // Load the rewrite space data
        let data = LoadedRewriteSpace::load_from_config(&config)?;

        // Use the centralized method to set selected commits
        let selected_commit = data.try_select_commits(&self.commits)?;

        // Update config with the selected commits
        config.set_selected_commits(&selected_commit)?;

        println!("Successfully selected {} commits:", selected_commit.len());
        display_commits(&selected_commit, &data.space, "");

        Ok(())
    }
}
