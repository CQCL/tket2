use std::fs;
use std::path::Path;

use clap::Parser;
use hugr::hugr::views::{HierarchyView, SiblingGraph};
use hugr::ops::handle::DfgID;
use hugr::HugrView;
use itertools::Itertools;

use tket2::json::load_tk1_json_file;
// Import the CircuitMatcher struct and its methods
use tket2::passes::taso::rep_sets_from_path;
use tket2::portmatching::{CircuitMatcher, CircuitPattern};

/// Simple program to greet a person
#[derive(Parser, Debug)]
#[clap(version = "1.0", long_about = None)]
#[clap(about = "Precompiles patterns from files into a CircuitMatcher stored as binary file.")]
struct CmdLineArgs {
    /// Name of input file/folder
    #[arg(
        short,
        long,
        value_name = "FILE",
        help = "Sets the input file or folder to use. It is either a JSON file of Quartz-generated ECCs or a folder with TK1 circuits in JSON format."
    )]
    input: String,
    /// Name of output file/folder
    #[arg(
        short,
        long,
        value_name = "FILE",
        default_value = ".",
        help = "Sets the output file or folder to use. Defaults to \"matcher.bin\" if no file name is provided."
    )]
    output: String,
}

fn main() {
    let opts = CmdLineArgs::parse();

    let input_path = Path::new(&opts.input);
    let output_path = Path::new(&opts.output);

    let all_circs = if input_path.is_file() {
        // Input is an ECC file in JSON format
        let eccs = rep_sets_from_path(input_path);
        eccs.into_iter()
            .flat_map(|ecc| ecc.into_circuits())
            .collect_vec()
    } else if input_path.is_dir() {
        // Input is a folder with TK1 circuits in JSON format
        fs::read_dir(input_path)
            .unwrap()
            .map(|file| {
                let path = file.unwrap().path();
                load_tk1_json_file(path).unwrap()
            })
            .collect_vec()
    } else {
        panic!("Input must be a file or a directory");
    };

    let patterns = all_circs
        .iter()
        .filter_map(|circ| {
            let circ: SiblingGraph<'_, DfgID> = SiblingGraph::new(&circ, circ.root());
            // Fail silently on empty or disconnected patterns
            CircuitPattern::try_from_circuit(&circ).ok()
        })
        .collect_vec();
    println!("Loaded {} patterns.", patterns.len());

    println!("Building matcher...");
    let output_file = if output_path.is_dir() {
        output_path.join("matcher.bin")
    } else {
        output_path.to_path_buf()
    };
    let matcher = CircuitMatcher::from_patterns(patterns);
    matcher.save_binary(output_file.to_str().unwrap()).unwrap();
    println!("Written matcher to {:?}", output_file);

    // Print the file size of output_file in megabytes
    if let Ok(metadata) = fs::metadata(&output_file) {
        let file_size = metadata.len() as f64 / (1024.0 * 1024.0);
        println!("File size: {:.2} MB", file_size);
    }
}
