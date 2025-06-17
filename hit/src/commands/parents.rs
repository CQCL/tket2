//! Parents command implementation.

use anyhow::Result;

use super::CommandExecutor;
use crate::config::Config;
use crate::display::display_commits;
use crate::storage::RewriteSpaceData;

#[derive(Debug)]
pub struct ParentsCommand {
    pub commit_id: String,
}

impl CommandExecutor for ParentsCommand {
    fn execute(&self) -> Result<()> {
        // Load the rewrite space data
        let config = Config::load_or_default()?;
        let data = RewriteSpaceData::load_from_config(&config)?;

        // Get the commit ID from the prefix
        let commit_id = data.get_commit_id(&self.commit_id)?;

        // Get parents of the commit
        let parents: Vec<_> = data.space.state_space.parents(commit_id).collect();

        if parents.is_empty() {
            println!("Commit {} has no parents.", commit_id);
            return Ok(());
        }

        display_commits(
            &data.space,
            &parents,
            &format!("Parents of commit {}:", commit_id),
        );
        println!();
        println!("Total parents: {}", parents.len());

        Ok(())
    }
}
