//! CLI tools for tket-hseries.

use clap::Parser;

/// CLI arguments.
#[derive(Parser, Debug)]
#[clap(version = "1.0", long_about = None)]
#[clap(about = "tket-hseries CLI tools.")]
#[group(id = "tket-hseries")]
#[non_exhaustive]
pub enum CliArgs {
    /// Generate serialized extensions.
    GenExtensions(hugr_cli::extensions::ExtArgs),
}
