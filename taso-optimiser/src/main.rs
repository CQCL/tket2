mod tracing;

use crate::tracing::Tracer;

use std::io::BufWriter;
use std::num::NonZeroUsize;
use std::path::PathBuf;
use std::process::exit;
use std::{fs, path::Path};

use clap::Parser;
use hugr::Hugr;
use tket2::optimiser::taso::log::TasoLogger;
use tket2::{
    json::{load_tk1_json_file, TKETDecode},
    optimiser::TasoOptimiser,
};
use tket_json_rs::circuit_json::SerialCircuit;

/// Optimise circuits using Quartz-generated ECCs.
///
/// Quartz: https://github.com/quantum-compiler/quartz
#[derive(Parser, Debug)]
#[clap(version = "1.0", long_about = None)]
#[clap(about = "Optimise circuits using Quartz-generated ECCs.")]
struct CmdLineArgs {
    /// Input circuit file as TK1 JSON.
    #[arg(
        short,
        long,
        value_name = "FILE",
        help = "Input. A quantum circuit in TK1 JSON format."
    )]
    input: PathBuf,
    /// Output circuit file
    #[arg(
        short,
        long,
        default_value = "out.json",
        value_name = "FILE",
        help = "Output. A quantum circuit in TK1 JSON format."
    )]
    output: PathBuf,
    /// ECC file
    #[arg(
        short,
        long,
        value_name = "ECC_FILE",
        help = "Sets the ECC file to use. It is a JSON file of Quartz-generated ECCs."
    )]
    eccs: PathBuf,
    /// Log output file
    #[arg(
        short,
        long,
        default_value = "taso-optimisation.log",
        value_name = "LOGFILE",
        help = "Logfile to to output the progress of the optimisation."
    )]
    logfile: Option<PathBuf>,
    /// Timeout in seconds (default=no timeout)
    #[arg(
        short,
        long,
        value_name = "TIMEOUT",
        help = "Timeout in seconds (default=None)."
    )]
    timeout: Option<u64>,
    /// Number of threads (default=1)
    #[arg(
        short = 'j',
        long,
        value_name = "N_THREADS",
        help = "The number of threads to use. By default, use a single thread."
    )]
    n_threads: Option<NonZeroUsize>,
}

fn save_tk1_json_file(path: impl AsRef<Path>, circ: &Hugr) -> Result<(), std::io::Error> {
    let file = fs::File::create(path)?;
    let writer = BufWriter::new(file);
    let serial_circ = SerialCircuit::encode(circ).unwrap();
    serde_json::to_writer_pretty(writer, &serial_circ)?;
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let opts = CmdLineArgs::parse();

    // Setup tracing subscribers for stdout and file logging.
    //
    // We need to keep the object around to keep the logging active.
    let _tracer = Tracer::setup_tracing(opts.logfile);

    let input_path = Path::new(&opts.input);
    let output_path = Path::new(&opts.output);
    let ecc_path = Path::new(&opts.eccs);

    let final_circ_json = fs::File::create("final_circ.json")?;
    let circ_candidates_csv = fs::File::create("best_circs.csv")?;
    let taso_logger = TasoLogger::new(final_circ_json, circ_candidates_csv);

    let circ = load_tk1_json_file(input_path)?;

    println!("Compiling rewriter...");
    let Ok(optimiser) = TasoOptimiser::default_with_eccs_json_file(ecc_path) else {
        eprintln!(
            "Unable to load ECC file {:?}. Is it a JSON file of Quartz-generated ECCs?",
            ecc_path
        );
        exit(1);
    };

    let n_threads = opts
        .n_threads
        // TODO: Default to multithreading once that produces better results.
        //.or_else(|| std::thread::available_parallelism().ok())
        .unwrap_or(NonZeroUsize::new(1).unwrap());
    println!("Using {n_threads} threads");

    println!("Optimising...");
    let opt_circ = optimiser.optimise_with_log(&circ, taso_logger, opts.timeout, n_threads);

    println!("Saving result");
    save_tk1_json_file(output_path, &opt_circ)?;

    println!("Done.");

    Ok(())
}
