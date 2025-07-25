//! CLI for tket-hseries

use anyhow::Result;
use clap::Parser as _;
use hugr::extension::ExtensionRegistry;
use tket_hseries::cli::CliArgs;

fn main() -> Result<()> {
    match CliArgs::parse() {
        CliArgs::GenExtensions(args) => {
            let reg = ExtensionRegistry::new([
                tket::extension::TKET_EXTENSION.to_owned(),
                tket::extension::rotation::ROTATION_EXTENSION.to_owned(),
                tket::extension::bool::BOOL_EXTENSION.to_owned(),
                tket::extension::debug::DEBUG_EXTENSION.to_owned(),
                tket::extension::guppy::GUPPY_EXTENSION.to_owned(),
                tket_hseries::extension::qsystem::EXTENSION.to_owned(),
                tket_hseries::extension::futures::EXTENSION.to_owned(),
                tket_hseries::extension::random::EXTENSION.to_owned(),
                tket_hseries::extension::result::EXTENSION.to_owned(),
                tket_hseries::extension::utils::EXTENSION.to_owned(),
                tket_hseries::extension::wasm::EXTENSION.to_owned(),
            ]);

            args.run_dump(&reg)?;
        }
        _ => {
            eprintln!("Unknown command");
            std::process::exit(1);
        }
    };

    Ok(())
}
