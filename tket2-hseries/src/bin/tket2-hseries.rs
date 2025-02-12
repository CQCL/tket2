//! CLI for tket2-hseries

use clap::Parser as _;
use hugr::extension::ExtensionRegistry;
use tket2_hseries::cli::CliArgs;

fn main() {
    match CliArgs::parse() {
        CliArgs::GenExtensions(args) => {
            let reg = ExtensionRegistry::new([
                tket2::extension::TKET2_EXTENSION.to_owned(),
                tket2::extension::rotation::ROTATION_EXTENSION.to_owned(),
                tket2_hseries::extension::qsystem::EXTENSION.to_owned(),
                tket2_hseries::extension::futures::EXTENSION.to_owned(),
                tket2_hseries::extension::result::EXTENSION.to_owned(),
                tket2_hseries::extension::wasm::EXTENSION.to_owned(),
                tket2_hseries::extension::utils::EXTENSION.to_owned(),
            ]);

            args.run_dump(&reg);
        }
        _ => {
            eprintln!("Unknown command");
            std::process::exit(1);
        }
    };
}
