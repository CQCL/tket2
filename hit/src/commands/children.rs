//! Children command implementation.

use anyhow::Result;
use itertools::Itertools;

use super::CommandExecutor;
use crate::config::Config;
use crate::display::display_commits;
use crate::storage::RewriteSpaceData;

#[derive(Debug)]
pub struct ChildrenCommand {
    pub commit_id: String,
}

impl CommandExecutor for ChildrenCommand {
    fn execute(&self) -> Result<()> {
        // Load the rewrite space data
        let config = Config::load_or_default()?;
        let data = RewriteSpaceData::load_from_config(&config)?;

        // Get the commit ID from the prefix
        let commit_id = data.get_commit_id(&self.commit_id)?;

        // Get children of the commit
        let children = data.space.state_space.children(commit_id).collect_vec();

        if children.is_empty() {
            println!("Commit {} has no children.", commit_id);
            return Ok(());
        }

        display_commits(
            &data.space,
            &children,
            &format!("Children of commit {}:", commit_id),
        );
        println!();
        println!("Total children: {}", children.len());

        Ok(())
    }
}
