use std::env;
use std::fs;
use std::path::PathBuf;

use hugr::hugr::views::DescendantsGraph;
use hugr::ops::handle::DfgID;
use hugr::Hugr;
use hugr::HugrView;
use tket2::circuit::HierarchyView;
use tket2::json::load_tk1_json_file;
use tket2::portmatching::CircuitPattern;

fn to_circ(h: &Hugr) -> DescendantsGraph<'_, DfgID> {
    DescendantsGraph::new(h, h.root())
}

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() != 2 {
        eprintln!("Usage: {} <folder>", args[0]);
        std::process::exit(1);
    }

    let folder = PathBuf::from(&args[1]);

    let mut files: Vec<PathBuf> = folder
        .read_dir()
        .expect("Failed to read directory")
        .map(|entry| entry.expect("Failed to read directory entry").path())
        .filter(|path| path.extension().map_or(false, |ext| ext == "json"))
        .collect();

    files.sort_unstable_by_key(|path| {
        let stem = path.file_stem().expect("invalid path");
        stem.to_str()
            .expect("not a valid path name")
            .parse::<usize>()
            .expect("file name not a number")
    });

    let mut index = 0;

    for path in files {
        let target_hugr = load_tk1_json_file(path.to_str().unwrap()).unwrap();
        let target_circ = to_circ(&target_hugr);

        if CircuitPattern::try_from_circuit(&target_circ).is_ok() {
            rename(path, index);
            index += 1;
        } else {
            remove(path);
        }
    }
}

fn remove(mut path: PathBuf) {
    fs::remove_file(&path).expect("Failed to remove file");
    println!("Removed file {}", path.display());
    path.set_extension("qasm");
    fs::remove_file(&path).expect("Failed to remove file");
    println!("Removed file {}", path.display());
}

fn rename(mut path: PathBuf, new_name: usize) {
    let mut new_path = if let Some(parent) = path.parent() {
        parent.join(format!("{}.json", new_name))
    } else {
        PathBuf::from(format!("{}.json", new_name))
    };
    fs::rename(&path, &new_path).expect("Failed to rename file");
    println!("Renamed file {} to {}", path.display(), new_path.display());
    // Also rename QASM files
    path.set_extension("qasm");
    new_path.set_extension("qasm");
    fs::rename(&path, &new_path).expect("Failed to rename file");
    println!("Renamed file {} to {}", path.display(), new_path.display());
}
