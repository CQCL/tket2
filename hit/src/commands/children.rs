//! Children command implementation.

use anyhow::Result;
use itertools::Itertools;

use super::CommandExecutor;
use crate::config::Config;
use crate::display::{display_commits, CommitHexId};
use crate::storage::LoadedRewriteSpace;

#[derive(Debug)]
pub struct ChildrenCommand {
    pub commit_id: String,
}

impl CommandExecutor for ChildrenCommand {
    fn execute(&self) -> Result<()> {
        // Load the rewrite space data
        let config = Config::load_or_default()?;
        let data = LoadedRewriteSpace::load_from_config(&config)?;

        // Get the commit ID from the prefix
        let commit = data.get_commit_from_hex(&self.commit_id)?;

        // Get children of the commit
        let children = commit.children(data.space.state_space()).collect_vec();

        if children.is_empty() {
            println!("Commit {} has no children.", CommitHexId(commit.id()));
            return Ok(());
        }

        display_commits(
            &children,
            &data.space,
            &format!("Children of commit {}:", CommitHexId(commit.id())),
        );
        println!();
        println!("Total children: {}", children.len());

        Ok(())
    }
}
