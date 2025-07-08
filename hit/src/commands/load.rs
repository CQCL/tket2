//! Load command implementation.

use std::path::{self, PathBuf};

use anyhow::{Context, Result};

use super::CommandExecutor;
use crate::config::Config;
use crate::storage::LoadedRewriteSpace;

#[derive(Debug)]
pub struct LoadCommand {
    pub filepath: PathBuf,
}

impl CommandExecutor for LoadCommand {
    fn execute(&self) -> Result<()> {
        // Verify the file exists by trying to load it
        let data = LoadedRewriteSpace::load(&self.filepath).with_context(|| {
            format!(
                "Failed to load rewrite space from {}",
                self.filepath.display()
            )
        })?;

        // Update config
        let mut config = Config::load_or_default()?;
        config.current_file = Some(path::absolute(&self.filepath)?);
        config.save()?;

        println!("Loaded rewrite space from {}", self.filepath.display());
        println!("Number of commits: {}", data.num_commits());
        Ok(())
    }
}
