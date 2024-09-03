use clap::Parser as _;
use clap_verbosity_flag::Level;
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
        _ => {
            eprintln!("Unknown command");
            std::process::exit(1);
        }
    };
}
