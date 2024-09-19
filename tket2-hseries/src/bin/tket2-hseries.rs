//! CLI for tket2-hseries

use std::process::ExitCode;

use clap::Parser as _;
use clap_verbosity_flag::Level;
use hugr::extension::ExtensionRegistry;
use tket2_hseries::cli::CliArgs;

fn main() -> ExitCode {
    match CliArgs::parse() {
        CliArgs::GenExtensions(args) => {
            let reg = ExtensionRegistry::try_new([
                tket2::extension::TKET2_EXTENSION.to_owned(),
                tket2::extension::rotation::ROTATION_EXTENSION.to_owned(),
                tket2_hseries::extension::hseries::EXTENSION.to_owned(),
                tket2_hseries::extension::futures::EXTENSION.to_owned(),
                tket2_hseries::extension::result::EXTENSION.to_owned(),
            ])
            .unwrap();

            args.run_dump(&reg);
            ExitCode::SUCCESS
        }
        CliArgs::Run(mut args) => {
            if let Err(e) = args.run() {
                if args.verbosity(Level::Error) {
                    eprintln!("Error: {}", e);
                }
                ExitCode::FAILURE
            } else {
                ExitCode::SUCCESS
            }
        }
        _ => {
            eprintln!("Unknown command");
            ExitCode::FAILURE
        }
    }
}
