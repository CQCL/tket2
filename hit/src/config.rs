//! Configuration management for hit CLI.

use anyhow::{Context, Result};
use hugr::persistent::Commit;
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::{fs, path::PathBuf};

use crate::constants::CONFIG_FILE;
use crate::display::CommitHexId;

#[derive(Serialize, Deserialize, Debug, Default)]
pub struct Config {
    /// Currently loaded rewrite space file
    pub current_file: Option<PathBuf>,
    /// Currently selected commits
    pub selected_commits: Vec<String>,
}

impl Config {
    /// Load config from file or return default if file doesn't exist
    pub fn load_or_default() -> Result<Self> {
        let config_path = Path::new(CONFIG_FILE);

        if !config_path.exists() {
            return Ok(Self::default());
        }

        let content = fs::read_to_string(&config_path)
            .with_context(|| format!("Failed to read config from {}", config_path.display()))?;

        let config: Config = toml::from_str(&content)
            .with_context(|| format!("Failed to parse config from {}", config_path.display()))?;

        Ok(config)
    }

    /// Save config to file
    pub fn save(&self) -> Result<()> {
        let config_path = Path::new(CONFIG_FILE);
        let content = toml::to_string_pretty(self).context("Failed to serialize config")?;

        fs::write(&config_path, content)
            .with_context(|| format!("Failed to write config to {}", config_path.display()))?;

        Ok(())
    }

    pub(crate) fn set_selected_commits(&mut self, selected_commit: &[Commit]) -> Result<()> {
        self.selected_commits = selected_commit
            .iter()
            .map(|c| format!("{}", CommitHexId(c.id())))
            .collect();
        self.save()
    }
}
