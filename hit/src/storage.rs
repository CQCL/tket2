//! Storage functionality for RewriteSpace data.

use anyhow::{anyhow, Context, Result};
use chrono::{DateTime, Utc};
use hugr_core::hugr::persistent::{CommitId, PersistentHugr};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;
use tket2::rewrite_space::RewriteSpace;

use crate::constants::HITFILES_DIR;

#[derive(Serialize, Deserialize, Debug)]
pub struct RewriteSpaceData {
    pub timestamp: DateTime<Utc>,
    pub space: RewriteSpace<isize>,
    /// Selected commits, always compatible and minimal (no commit is an
    /// ancestor of another)
    pub selected_commits: Vec<CommitId>,
}

impl RewriteSpaceData {
    /// Create a new RewriteSpaceData from a RewriteSpace
    pub fn new(space: RewriteSpace<isize>) -> Result<Self> {
        let timestamp = Utc::now();

        // Select only the base commit by default
        let selected_commits = vec![space.base()];

        Ok(Self {
            timestamp,
            space,
            selected_commits,
        })
    }

    /// Get the current PersistentHugr from the selected commits
    pub fn current_hugr(&self) -> PersistentHugr {
        self.space
            .state_space
            .try_extract_hugr(self.selected_commits.iter().copied())
            .expect("Selected commits are always compatible by invariant")
    }

    /// Set selected commits by matching prefixes, ensuring compatibility and
    /// minimality
    pub fn set_selected_commits(&mut self, commit_prefixes: &[String]) -> Result<()> {
        if commit_prefixes.is_empty() {
            // Select only the base commit
            let base_commit_id = self.space.base();

            self.selected_commits = vec![base_commit_id];
            return Ok(());
        }

        // Find commits matching the prefixes
        let all_commits: Vec<CommitId> = self.space.all_commit_ids().collect();
        let mut matched_commits = Vec::new();

        for prefix in commit_prefixes {
            let matching_commits: Vec<_> = all_commits
                .iter()
                .filter(|commit| format!("{}", commit).starts_with(prefix))
                .collect();

            match matching_commits.len() {
                0 => return Err(anyhow!("No commit found matching prefix: {}", prefix)),
                1 => matched_commits.push(*matching_commits[0]),
                _ => {
                    let matches: Vec<String> =
                        matching_commits.iter().map(|c| format!("{}", c)).collect();
                    return Err(anyhow!(
                        "Multiple commits match prefix '{}': {}. Please be more specific.",
                        prefix,
                        matches.join(", ")
                    ));
                }
            }
        }

        // Check if commits are compatible by trying to extract
        let hugr = self
            .space
            .state_space
            .try_extract_hugr(matched_commits.iter().copied())
            .context("Selected commits are not compatible")?;

        // Find minimal subset (commits with no children among the selected)
        let minimal_commits = self.find_minimal_commit_subset(&hugr);

        self.selected_commits = minimal_commits;
        Ok(())
    }

    /// Find the minimal subset of commits (those that have no children among
    /// the selected commits)
    fn find_minimal_commit_subset(&self, hugr: &PersistentHugr) -> Vec<CommitId> {
        hugr.all_commit_ids()
            .filter(|&commit| hugr.as_state_space().children(commit).next().is_none())
            .collect()
    }

    /// Get the currently selected commit IDs
    pub fn get_selected_commit_ids(&self) -> Vec<CommitId> {
        self.selected_commits.clone()
    }

    /// Save this RewriteSpaceData to a file with timestamp-based name
    pub fn save(&self) -> Result<String> {
        fs::create_dir_all(HITFILES_DIR)?;

        let filename = format!("{}.json", self.timestamp.format("%Y%m%d_%H%M%S"));
        let filepath = Path::new(HITFILES_DIR).join(&filename);

        let json =
            serde_json::to_string_pretty(self).context("Failed to serialize rewrite space data")?;

        fs::write(&filepath, json)
            .with_context(|| format!("Failed to write to {}", filepath.display()))?;

        Ok(filename)
    }

    /// Load RewriteSpaceData from a file
    pub fn load(filename: &str) -> Result<Self> {
        let filepath = Path::new(HITFILES_DIR).join(filename);

        let json_content = fs::read_to_string(&filepath)
            .with_context(|| format!("Failed to read {}", filepath.display()))?;

        let data: RewriteSpaceData =
            serde_json::from_str(&json_content).context("Failed to parse rewrite space data")?;

        Ok(data)
    }

    /// Get the number of commits in the space
    pub fn num_commits(&self) -> usize {
        self.space.all_commit_ids().count()
    }

    /// Find the latest file in the hitfiles directory
    pub fn find_latest_file() -> Result<String> {
        let entries = fs::read_dir(HITFILES_DIR).context("Failed to read .hitfiles directory")?;

        let json_files = entries.filter_map(|entry| {
            let entry = entry.ok()?;
            let path = entry.path();
            if path.extension()? == "json" {
                Some(path.file_name()?.to_string_lossy().to_string())
            } else {
                None
            }
        });

        json_files
            .into_iter()
            .max()
            .ok_or_else(|| anyhow::anyhow!("No JSON files found in .hitfiles directory"))
    }
}
