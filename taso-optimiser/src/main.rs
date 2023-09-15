use std::{fs, io, path::Path};

use clap::Parser;
use hugr::Hugr;
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
    input: String,
    /// Output circuit file
    #[arg(
        short,
        long,
        default_value = "out.json",
        value_name = "FILE",
        help = "Output. A quantum circuit in TK1 JSON format."
    )]
    output: String,
    /// ECC file
    #[arg(
        short,
        long,
        value_name = "ECC_FILE",
        help = "Sets the ECC file to use. It is a JSON file of Quartz-generated ECCs."
    )]
    eccs: String,
    /// Timeout in seconds (default=100)
    #[arg(
        short,
        long,
        value_name = "TIMEOUT",
        help = "Timeout in seconds (default=None)."
    )]
    timeout: Option<u64>,
    /// Number of threads (default=1)
    #[arg(
        short,
        long,
        default_value = "1",
        value_name = "N_THREADS",
        help = "The number of threads to use. Currently only single-threaded TASO is supported."
    )]
    n_threads: usize,
}

fn save_tk1_json_file(path: impl AsRef<Path>, circ: &Hugr) -> Result<(), std::io::Error> {
    let file = fs::File::create(path)?;
    let writer = io::BufWriter::new(file);
    let serial_circ = SerialCircuit::encode(circ).unwrap();
    serde_json::to_writer_pretty(writer, &serial_circ)?;
    Ok(())
}

fn main() {
    let opts = CmdLineArgs::parse();

    let input_path = Path::new(&opts.input);
    let output_path = Path::new(&opts.output);
    let ecc_path = Path::new(&opts.eccs);

    let circ = load_tk1_json_file(input_path).unwrap();

    println!("Compiling rewriter...");
    let optimiser = if opts.n_threads == 1 {
        println!("Using single-threaded TASO");
        TasoOptimiser::default_with_eccs_json_file(ecc_path)
    } else {
        unimplemented!("Multi-threaded TASO has been disabled until fixed");
    };
    println!("Optimising...");
    let opt_circ = optimiser
        .optimise_with_default_log(&circ, opts.timeout)
        .unwrap();

    println!("Saving result");
    save_tk1_json_file(output_path, &opt_circ).unwrap();

    println!("Done.")
}
