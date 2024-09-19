//! CLI tools for tket2-hseries.

use std::io::Write;
use clap::Parser;
use clap_verbosity_flag::Level;
use clio::Output;
use hugr::{algorithms::validation::ValidationLevel, Hugr};
use hugr_cli::HugrArgs;

use crate::{HSeriesPass, HSeriesPassError};

/// CLI arguments.
#[derive(Parser, Debug)]
#[clap(version = "1.0", long_about = None)]
#[clap(about = "tket2-hseries CLI tools.")]
#[group(id = "tket2-hseries")]
#[non_exhaustive]
pub enum CliArgs {
    /// Generate serialized extensions.
    GenExtensions(hugr_cli::extensions::ExtArgs),
    Run(RunCliArgs),
}

#[derive(Parser, Debug)]
pub struct RunCliArgs {
    #[clap(long, short, value_parser, default_value = "-")]
    output: Output,

    #[command(flatten)]
    pub run_args: RunArgs,
}

#[derive(Parser, Debug)]
pub struct RunArgs {
    #[clap(long="tket2", default_value = "true")]
    pub run_lowertket2: bool,

    #[clap(long="lazify", default_value = "true")]
    pub run_lazify_measure: bool,

    #[clap(long="order", default_value = "true")]
    pub run_force_order: bool,

    #[clap(long, default_value = "true")]
    pub validate: bool,

    #[command(flatten)]
    pub hugr_args: HugrArgs,
}

impl RunArgs {
    pub fn run(&mut self) -> Result<Hugr, Box<dyn std::error::Error>> {
        let (hugrs, registry) = self.hugr_args.validate()?;
        let [mut hugr] = hugrs.try_into().map_err(|hugrs: Vec<Hugr>| format!("Expected exactly one HUGR, found {}", hugrs.len()))?;

        let level = if self.validate {
            ValidationLevel::WithoutExtensions
        } else {
            ValidationLevel::None
        };
        let pass = HSeriesPass::default().with_validation_level(level).with_lowertket2(self.run_lowertket2).with_lazify_measure(self.run_lazify_measure).with_force_order(self.run_force_order);

        pass.run(&mut hugr, &registry)?;
        Ok(hugr)
    }
}

impl RunCliArgs {
    pub fn run(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        write!(self.output, "{}", serde_json::to_string_pretty(&self.run_args.run()?)?)?;
        Ok(())
    }

    /// Test whether a `level` message should be output.
    pub fn verbosity(&self, level: Level) -> bool {
        self.run_args.hugr_args.verbosity(level)
    }
}
