//! Run command implementation.

use anyhow::{anyhow, Context, Result};
use hugr::ops::OpType;
use std::fs;
use std::io::BufReader;
use std::path::PathBuf;
use tket2::circuit::cost::is_cx;
use tket2::rewrite_space::{BoxedCommitFactory, ExploreOptions, RewriteSpace};
use tket2::Circuit;

use super::CommandExecutor;
use crate::config::Config;
use crate::factory::CommuteCZFactory;
use crate::storage::RewriteSpaceData;

#[derive(Debug)]
pub struct RunCommand {
    pub input_file: PathBuf,
    pub factory: String,
    pub max_rewrites: Option<usize>,
}

fn cost_fn(op: &OpType) -> usize {
    is_cx(op) as usize
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

        // Set up commit factory exploration options
        let opts = ExploreOptions {
            max_rewrites: self.max_rewrites,
            pattern_radius: 4,
            cost_fn,
        };

        // Run explorer based on type
        let factory: BoxedCommitFactory = match self.factory.as_str() {
            "CommuteCZ" => {
                println!("Running CommuteCZ commit factory...");
                CommuteCZFactory.into()
            }
            // Add further explorers here
            _ => {
                return Err(anyhow!(
                    "Unknown explorer type: {}. Currently only 'CommuteCZ' is supported.",
                    self.factory
                ));
            }
        };
        space.explore(factory, &opts);

        let commit_count = space.all_commit_ids().count();
        println!(
            "Exploration complete. Found {} total commits.",
            commit_count
        );

        // Create and save rewrite space data
        let data = RewriteSpaceData::with_current_time(space)?;
        let filename = data.save()?;

        println!("Saved rewrite space to {}", filename);

        // Update config to point to this file
        let mut config = Config::load_or_default()?;
        config.current_file = Some(filename);
        config.selected_commits = vec![data.space.base().to_string()];
        config.save()?;

        Ok(())
    }
}
