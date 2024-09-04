use clap::Parser as _;
use clap_verbosity_flag::Level;
use hugr::extension::ExtensionRegistry;
use tket2_hseries::cli::CliArgs;

fn main() {
    match CliArgs::parse() {
        CliArgs::Run(mut args) => {
            if let Err(e) = args.run() {
                if args.hugr_args.verbosity(Level::Error) {
                    eprintln!("{}", e);
                }
            }
        }
        CliArgs::GenExtensions(args) => {
            let reg = ExtensionRegistry::try_new([
                tket2::extension::TKET2_EXTENSION.to_owned(),
                tket2_hseries::extension::hseries::EXTENSION.to_owned(),
                tket2_hseries::extension::futures::EXTENSION.to_owned(),
                tket2_hseries::extension::result::EXTENSION.to_owned(),
            ])
            .unwrap();

            args.run_dump(&reg);
        }
        _ => {
            eprintln!("Unknown command");
            std::process::exit(1);
        }
    };
}
