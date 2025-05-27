//! Show command implementation.

use anyhow::{anyhow, Context, Result};
use hugr::HugrView;

use super::CommandExecutor;
use crate::config::Config;
use crate::storage::RewriteSpaceData;

#[derive(Debug)]
pub struct ShowCommand;

impl CommandExecutor for ShowCommand {
    fn execute(&self) -> Result<()> {
        let config = Config::load_or_default()?;

        let current_file = config
            .current_file
            .as_ref()
            .ok_or_else(|| anyhow!("No current file loaded. Use 'hit load' first."))?;

        // Load the rewrite space data
        let data = RewriteSpaceData::load(current_file)
            .with_context(|| format!("Failed to load rewrite space from {}", current_file))?;

        // Get the current HUGR from selected commits
        let current_hugr = data.current_hugr();
        let hugr = current_hugr.to_hugr();

        // Print the mermaid string
        println!("{}", hugr.mermaid_string());

        Ok(())
    }
}
