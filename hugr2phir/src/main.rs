mod normalize;

use std::fs::File;
use std::io::BufReader;
use std::path::PathBuf;

use clap::Parser;

use hugr::{
    hugr::views::{DescendantsGraph, HierarchyView},
    ops::{OpTag, OpTrait, OpType},
    Hugr, HugrView,
};

use tket2::phir::circuit_to_phir;

#[derive(Parser, Debug)]
#[clap(version = "1.0", long_about = None)]
#[clap(about = "Convert from hugr msgpack serialized form to PHIR JSON.")]
#[command(long_about = "Sets the input file to use. It must be serialized HUGR.")]
struct CmdLineArgs {
    /// Name of input file/folder
    input: PathBuf,
    /// Name of output file/folder
    #[arg(
        short,
        long,
        value_name = "FILE",
        default_value = None,
        help = "Sets the output file or folder. Defaults to the same as the input file with a .json extension."
    )]
    output: Option<PathBuf>,
}

fn main() {
    let CmdLineArgs { input, output } = CmdLineArgs::parse();

    let reader = BufReader::new(File::open(&input).unwrap());
    let output = output.unwrap_or_else(|| {
        let mut output = input.clone();
        output.set_extension("json");
        output
    });

    let mut hugr: Hugr = rmp_serde::from_read(reader).unwrap();
    normalize::remove_identity_tuples(&mut hugr);
    // DescendantsGraph::try_new(&hugr, root).unwrap()
    let root = hugr.root();
    let root_op_tag = hugr.get_optype(root).tag();
    let circ: DescendantsGraph = if OpTag::DataflowParent.is_superset(root_op_tag) {
        // Some dataflow graph
        DescendantsGraph::try_new(&hugr, root).unwrap()
    } else if OpTag::ModuleRoot.is_superset(root_op_tag) {
        // Assume Guppy generated module

        // just take the first function
        let main_node = hugr
            .children(hugr.root())
            .find(|n| matches!(hugr.get_optype(*n), OpType::FuncDefn(_)))
            .expect("Module contains no functions.");
        // just take the first node again...assume guppy source so always top
        // level CFG
        let cfg_node = hugr
            .children(main_node)
            .find(|n| matches!(hugr.get_optype(*n), OpType::CFG(_)))
            .expect("Function contains no cfg.");

        // Now is a bit sketchy...assume only one basic block in CFG
        let block_node = hugr
            .children(cfg_node)
            .find(|n| matches!(hugr.get_optype(*n), OpType::BasicBlock(_)))
            .expect("CFG contains no basic block.");
        DescendantsGraph::try_new(&hugr, block_node).unwrap()
    } else {
        panic!("HUGR Root Op type {root_op_tag:?} not supported");
    };

    let phir = circuit_to_phir(&circ).unwrap();

    serde_json::to_writer(File::create(output).unwrap(), &phir).unwrap();
}
