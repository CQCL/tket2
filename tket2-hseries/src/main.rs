//! A command line interface to tket2-hseries
use hugr_cli::{Level, Parser as _};
use tket2_hseries::cli;

fn main() {
    let opts = cli::CmdLineArgs::parse();
    let registry = &cli::REGISTRY;

    // validate with all std extensions
    if let Err(e) = opts.run(registry) {
        if opts.verbosity(Level::Error) {
            eprintln!("{}", e);
        }
        std::process::exit(1);
    }
}
