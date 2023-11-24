mod tracing;

use crate::tracing::Tracer;

use std::fs::File;
use std::io::BufWriter;
use std::num::NonZeroUsize;
use std::path::Path;
use std::path::PathBuf;
use std::process::exit;

use clap::Parser;
use tket2::json::{load_tk1_json_file, save_tk1_json_file};
use tket2::optimiser::badger::log::BadgerLogger;
use tket2::optimiser::badger::BadgerOptions;
use tket2::optimiser::BadgerOptimiser;

#[cfg(feature = "peak_alloc")]
#[global_allocator]
static PEAK_ALLOC: peak_alloc::PeakAlloc = peak_alloc::PeakAlloc;

#[cfg(all(not(target_env = "msvc"), not(feature = "peak_alloc")))]
#[global_allocator]
static GLOBAL: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;

/// Optimise circuits using Quartz-generated ECCs.
///
/// Quartz: <https://github.com/quantum-compiler/quartz>
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
        default_value = "badger-optimisation.log",
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
    /// Maximum time in seconds to wait between circuit improvements (default=no timeout)
    #[arg(
        short = 'p',
        long,
        value_name = "PROGRESS_TIMEOUT",
        help = "Maximum time in seconds to wait between circuit improvements (default=None)."
    )]
    progress_timeout: Option<u64>,
    /// Number of threads (default=1)
    #[arg(
        short = 'j',
        long,
        value_name = "N_THREADS",
        help = "The number of threads to use. By default, use a single thread."
    )]
    n_threads: Option<NonZeroUsize>,
    /// Split the circuit into chunks, and process them separately.
    #[arg(
        long = "split-circ",
        help = "Split the circuit into chunks and optimize each one in a separate thread. Use `-j` to specify the number of threads to use."
    )]
    split_circ: bool,
    /// Max queue size.
    #[arg(
        short = 'q',
        long = "queue-size",
        default_value = "100",
        value_name = "QUEUE_SIZE",
        help = "The priority queue size. Defaults to 100."
    )]
    queue_size: usize,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let opts = CmdLineArgs::parse();

    let input_path = Path::new(&opts.input);
    let output_path = Path::new(&opts.output);
    let ecc_path = Path::new(&opts.eccs);

    let n_threads = opts
        .n_threads
        // TODO: Default to multithreading once that produces better results.
        //.or_else(|| std::thread::available_parallelism().ok())
        .unwrap_or(NonZeroUsize::new(1).unwrap());

    // Setup tracing subscribers for stdout and file logging.
    //
    // We need to keep the object around to keep the logging active.
    let _tracer = Tracer::setup_tracing(opts.logfile, n_threads.get() > 1);

    // TODO: Remove this from the Logger, and use tracing events instead.
    let circ_candidates_csv = BufWriter::new(File::create("best_circs.csv")?);

    let badger_logger = BadgerLogger::new(circ_candidates_csv);

    let circ = load_tk1_json_file(input_path)?;

    println!("Compiling rewriter...");
    let Ok(optimiser) = BadgerOptimiser::default_with_eccs_json_file(ecc_path) else {
        eprintln!(
            "Unable to load ECC file {:?}. Is it a JSON file of Quartz-generated ECCs?",
            ecc_path
        );
        exit(1);
    };
    println!(
        "Using {n_threads} threads. Queue size is {}.",
        opts.queue_size
    );

    if opts.split_circ && n_threads.get() > 1 {
        println!("Splitting circuit into {n_threads} chunks.");
    }

    println!("Optimising...");
    let opt_circ = optimiser.optimise_with_log(
        &circ,
        badger_logger,
        BadgerOptions {
            timeout: opts.timeout,
            progress_timeout: opts.progress_timeout,
            n_threads,
            split_circuit: opts.split_circ,
            queue_size: opts.queue_size,
        },
    );

    println!("Saving result");
    save_tk1_json_file(&opt_circ, output_path)?;

    #[cfg(feature = "peak_alloc")]
    println!("Peak memory usage: {} GB", PEAK_ALLOC.peak_usage_as_gb());

    println!("Done.");
    Ok(())
}
