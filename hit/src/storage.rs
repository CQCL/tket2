//! Storage functionality for RewriteSpace data.

use anyhow::{anyhow, Context, Result};
use hugr::persistent::{Commit, CommitId, PersistentHugr};
use itertools::Itertools;
use std::fs;
use std::path::{Path, PathBuf};
use tket::circuit::cost::LexicographicCost;
use tket::rewrite_space::{RewriteSpace, SerialRewriteSpace};

use crate::config::Config;
use crate::display::CommitHexId;

type CommitCost = tket::optimiser::seadog::Cost<LexicographicCost<usize, 2>>;

#[derive(Debug, Clone)]
pub struct LoadedRewriteSpace {
    pub filepath: PathBuf,
    pub space: RewriteSpace<CommitCost>,
    hugrs: Vec<PersistentHugr>,
    // Lifetime: we know the commits are valid as long as the rewrite space is
    // (when returning them to the user, we must adjust lifetime to self)
    all_commits: Vec<Commit<'static>>,
}

impl LoadedRewriteSpace {
    /// Load RewriteSpaceData from the current config
    pub fn load_from_config(config: &Config) -> Result<Self> {
        let current_file = config
            .current_file
            .as_ref()
            .ok_or_else(|| anyhow!("No current file loaded. Use 'hit load' first."))?;

        Self::load(current_file)
    }

    pub fn get_commit(&self, id: CommitId) -> Commit<'_> {
        unsafe {
            self.all_commits
                .iter()
                .find(|c| c.id() == id)
                .unwrap()
                .clone()
                .upgrade_lifetime()
        }
    }

    /// Get a commit ID from a prefix string
    pub fn get_commit_from_hex(&self, prefix: &str) -> Result<Commit<'_>> {
        let matching_commits = self
            .all_commits
            .iter()
            .filter(|commit| format!("{}", CommitHexId(commit.id())).starts_with(prefix));

        match matching_commits.at_most_one() {
            Ok(Some(commit)) => Ok(commit.clone()),
            Ok(None) => Err(anyhow!("No commit found matching prefix: {}", prefix)),
            Err(err) => {
                let matches: Vec<String> =
                    err.map(|c| format!("{}", CommitHexId(c.id()))).collect();
                Err(anyhow!(
                    "Multiple commits match prefix '{}': {}. Please be more specific.",
                    prefix,
                    matches.join(", ")
                ))
            }
        }
    }

    pub fn get_all_commits(&self) -> impl Iterator<Item = Commit<'_>> {
        self.all_commits
            .iter()
            .map(|c| unsafe { c.clone().upgrade_lifetime() })
    }

    /// Get the current PersistentHugr from the selected commits
    pub fn current_hugr(&self, config: &Config) -> Result<PersistentHugr> {
        let selected_commits = config
            .selected_commits
            .iter()
            .map(|c| {
                self.get_commit_from_hex(c)
                    .context("Invalid commit ID in config")
            })
            .collect::<Result<Vec<_>>>()?;

        PersistentHugr::try_new(selected_commits).context("Selected commits are not compatible")
    }

    /// Set selected commits by matching prefixes, ensuring compatibility and
    /// minimality
    pub fn try_select_commits(&self, commit_prefixes: &[String]) -> Result<Vec<Commit<'_>>> {
        if commit_prefixes.is_empty() {
            return Ok(vec![self.space.state_space().base_commit().ok_or(
                anyhow!("Could not select any commit, empty rewrite space"),
            )?]);
        }

        // Find commits matching the prefixes
        let mut matched_commits = Vec::new();

        for prefix in commit_prefixes {
            matched_commits.push(self.get_commit_from_hex(prefix)?);
        }

        // Check if commits are compatible by trying to extract
        let hugr = PersistentHugr::try_new(matched_commits)
            .context("Selected commits are not compatible")?;

        // Find minimal subset (commits with no children among the selected)
        Ok(self.find_minimal_commit_subset(&hugr))
    }

    /// Find the minimal subset of commits (those that have no children among
    /// the selected commits)
    fn find_minimal_commit_subset(&self, hugr: &PersistentHugr) -> Vec<Commit<'_>> {
        // necessary to be able to upgrade commit lifetime
        debug_assert_eq!(self.space.state_space(), hugr.state_space());

        hugr.all_commit_ids()
            .filter(|&commit| hugr.deleted_nodes(commit).next().is_none())
            .map(|id| unsafe { hugr.get_commit(id).clone().upgrade_lifetime() })
            .collect()
    }

    /// Load RewriteSpaceData from a file
    pub fn load(filepath: impl AsRef<Path>) -> Result<Self> {
        let f = fs::File::open(&filepath)
            .with_context(|| format!("Failed to read {}", filepath.as_ref().display()))?;

        let serial_space: SerialRewriteSpace<CommitCost> = serde_json::from_reader(f).context(
            format!("Failed to parse JSON from {}", filepath.as_ref().display()),
        )?;

        let mut space = RewriteSpace::new();
        let hugrs = space.replace_with(serial_space);
        let all_commits = hugrs
            .iter()
            .flat_map(|hugr| {
                hugr.all_commit_ids().map(|id| {
                    let commit = hugr.get_commit(id).clone();
                    unsafe { commit.upgrade_lifetime() }
                })
            })
            .unique_by(|commit| commit.id())
            .collect();

        let data = LoadedRewriteSpace {
            filepath: filepath.as_ref().to_path_buf(),
            space,
            hugrs,
            all_commits,
        };

        Ok(data)
    }

    /// Get the number of commits in the space
    #[allow(unused)]
    pub fn num_commits(&self) -> usize {
        self.all_commits.len()
    }
}
