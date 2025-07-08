//! Show command implementation.

use anyhow::Result;
use hugr::HugrView;

use super::CommandExecutor;
use crate::config::Config;
use crate::storage::RewriteSpaceData;

#[derive(Debug)]
pub struct ShowCommand;

impl CommandExecutor for ShowCommand {
    fn execute(&self) -> Result<()> {
        // Load the rewrite space data
        let config = Config::load_or_default()?;
        let data = RewriteSpaceData::load_from_config(&config)?;

        // Get the current HUGR from selected commits
        let current_hugr = data.current_hugr(&config)?;

        // Print the mermaid string
        println!("{}", current_hugr.mermaid_string());

        Ok(())
    }
}
