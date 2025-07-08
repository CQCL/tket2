//! Parents command implementation.

use anyhow::Result;

use super::CommandExecutor;
use crate::config::Config;
use crate::display::{display_commits, CommitHexId};
use crate::storage::LoadedRewriteSpace;

#[derive(Debug)]
pub struct ParentsCommand {
    pub commit_id: String,
}

impl CommandExecutor for ParentsCommand {
    fn execute(&self) -> Result<()> {
        // Load the rewrite space data
        let config = Config::load_or_default()?;
        let data = LoadedRewriteSpace::load_from_config(&config)?;

        // Get the commit ID from the prefix
        let commit = data.get_commit_from_hex(&self.commit_id)?;

        // Get parents of the commit
        let parents: Vec<_> = commit.parents().collect();

        if parents.is_empty() {
            println!("Commit {} has no parents.", CommitHexId(commit.id()));
            return Ok(());
        }

        display_commits(
            parents.iter().copied(),
            &data.space,
            &format!("Parents of commit {}:", CommitHexId(commit.id())),
        );
        println!();
        println!("Total parents: {}", parents.len());

        Ok(())
    }
}
