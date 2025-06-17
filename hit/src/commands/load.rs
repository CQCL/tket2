//! Load command implementation.

use anyhow::{Context, Result};

use super::CommandExecutor;
use crate::config::Config;
use crate::storage::RewriteSpaceData;

#[derive(Debug)]
pub struct LoadCommand {
    pub timestamp: Option<String>,
}

impl CommandExecutor for LoadCommand {
    fn execute(&self) -> Result<()> {
        let filename = match &self.timestamp {
            Some(ts) => format!("{}.json", ts),
            None => RewriteSpaceData::find_latest_file()?,
        };

        // Verify the file exists by trying to load it
        let _data = RewriteSpaceData::load(&filename)
            .with_context(|| format!("Failed to load rewrite space from {}", filename))?;

        // Update config
        let mut config = Config::load_or_default()?;
        config.current_file = Some(filename.clone());
        config.save()?;

        println!("Loaded rewrite space from {}", filename);
        Ok(())
    }
}
