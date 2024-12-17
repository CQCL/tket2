//! CLI tools for tket2-hseries.

use clap::Parser;

/// CLI arguments.
#[derive(Parser, Debug)]
#[clap(version = "1.0", long_about = None)]
#[clap(about = "tket2-hseries CLI tools.")]
#[group(id = "tket2-hseries")]
#[non_exhaustive]
pub enum CliArgs {
    /// Generate serialized extensions.
    GenExtensions(tket2::hugr::cli::extensions::ExtArgs),
}
