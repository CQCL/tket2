//! Provides a command line interface to tket2-hseries
use clap::Parser;
use hugr::std_extensions::arithmetic::{
    conversions::EXTENSION as CONVERSIONS_EXTENSION, float_ops::EXTENSION as FLOAT_OPS_EXTENSION,
    float_types::EXTENSION as FLOAT_TYPES_EXTENSION, int_ops::EXTENSION as INT_OPS_EXTENSION,
    int_types::EXTENSION as INT_TYPES_EXTENSION,
};
use hugr::std_extensions::logic::EXTENSION as LOGICS_EXTENSION;

use hugr::extension::{ExtensionRegistry, PRELUDE};
use lazy_static::lazy_static;

lazy_static! {
    /// A registry suitable for passing to `run`. Use this unless you have a
    /// good reason not to do so.
    pub static ref REGISTRY: ExtensionRegistry = ExtensionRegistry::try_new([
        PRELUDE.to_owned(),
        INT_OPS_EXTENSION.to_owned(),
        INT_TYPES_EXTENSION.to_owned(),
        CONVERSIONS_EXTENSION.to_owned(),
        FLOAT_OPS_EXTENSION.to_owned(),
        FLOAT_TYPES_EXTENSION.to_owned(),
        LOGICS_EXTENSION.to_owned(),
    ])
    .unwrap();
}

/// Arguments for `run`.
#[derive(Parser, Debug)]
#[command(version, about)]
pub struct CmdLineArgs {
    #[command(flatten)]
    base: hugr_cli::CmdLineArgs,
}

impl CmdLineArgs {
    /// Run the ngrte preparation and validation workflow with the given
    /// registry.
    pub fn run(&self, registry: &ExtensionRegistry) -> Result<(), hugr_cli::CliError> {
        let mut hugr = self.base.run(registry)?;
        crate::prepare_ngrte(&mut hugr).unwrap();
        serde_json::to_writer_pretty(std::io::stdout(), &hugr)?;
        Ok(())
    }

    /// Test whether a `level` message should be output.
    pub fn verbosity(&self, level: hugr_cli::Level) -> bool {
        self.base.verbosity(level)
    }
}
