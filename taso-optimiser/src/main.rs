use std::{fs, io, path::Path};

use clap::Parser;
use hugr::Hugr;
use tket2::{
    circuit::Circuit,
    json::{load_tk1_json_file, TKETDecode},
    passes::taso::taso,
    rewrite::{strategy::ExhaustiveRewriteStrategy, ECCRewriter},
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
    let rewriter = ECCRewriter::from_eccs_json_file(ecc_path);
    let strategy = ExhaustiveRewriteStrategy::default();

    println!("Optimising...");
    let opt_circ = taso(circ, rewriter, strategy, |c| c.num_gates(), Some(100));

    println!("Saving result");
    save_tk1_json_file(output_path, &opt_circ).unwrap();

    println!("Done.")
}
