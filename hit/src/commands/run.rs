//! Run command implementation.

use anyhow::{anyhow, Context, Result};
use hugr::{Hugr, HugrView};
use std::fs;
use std::io::BufReader;
use std::path::PathBuf;
use tket2::rewrite_space::{Explore, ExploreOptions, RewriteSpace};
use tket2::Circuit;

use super::CommandExecutor;
use crate::config::Config;
use crate::explore::CommuteCZExplorer;
use crate::storage::RewriteSpaceData;

#[derive(Debug)]
pub struct RunCommand {
    pub input_file: PathBuf,
    pub explorer: String,
    pub max_rewrites: Option<usize>,
}

impl CommandExecutor for RunCommand {
    fn execute(&self) -> Result<()> {
        // Load Circuit from HUGR envelope format
        let file = fs::File::open(&self.input_file)
            .with_context(|| format!("Failed to open input file: {}", self.input_file.display()))?;
        let reader = BufReader::new(file);

        // Try to load as envelope format
        let circuit = Circuit::load(reader, None)
            .with_context(|| format!("Failed to load HUGR from {}", self.input_file.display()))?;

        println!(
            "Loaded circuit with {} operations",
            circuit.num_operations()
        );

        // Create rewrite space
        let mut space = RewriteSpace::with_base(circuit.into_hugr());

        // Set up explorer options
        let opts = ExploreOptions {
            max_rewrites: self.max_rewrites,
        };

        // Run explorer based on type
        match self.explorer.as_str() {
            "CommuteCZ" => {
                let explorer = CommuteCZExplorer;
                println!("Running CommuteCZ explorer...");
                explorer.explore(&mut space, &opts);
            }
            // Add further explorers here
            _ => {
                return Err(anyhow!(
                    "Unknown explorer type: {}. Currently only 'CommuteCZ' is supported.",
                    self.explorer
                ));
            }
        }

        let commit_count = space.all_commit_ids().count();
        println!(
            "Exploration complete. Found {} total commits.",
            commit_count
        );

        // Create and save rewrite space data
        let data = RewriteSpaceData::new(space)?;
        let filename = data.save()?;

        println!("Saved rewrite space to {}", filename);

        // Update config to point to this file
        let mut config = Config::load_or_default()?;
        config.current_file = Some(filename);
        config.save()?;

        Ok(())
    }
}
