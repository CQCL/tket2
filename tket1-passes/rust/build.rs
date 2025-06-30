//! Build script for the tket1-passes crate
//!
//! Use bindgen to generate Rust bindings for the C library.

use std::env;
use std::path;
use std::path::PathBuf;

const LIB_NAME: &str = "tket1-passes";

fn libtket1_passes_name() -> String {
    let target = env::var("TARGET").unwrap();
    match target.as_str() {
        t if t.contains("windows") => format!("{LIB_NAME}.lib"),
        t if t.contains("apple") => format!("lib{LIB_NAME}.dylib"),
        _ => format!("lib{LIB_NAME}.so"),
    }
}

fn is_macos() -> bool {
    let target = env::var("TARGET").unwrap();
    target.contains("apple")
}

fn main() {
    let lib_path = PathBuf::from(env::var("TKET_LIB_PATH").unwrap_or("../lib".to_string()))
        .join(libtket1_passes_name());
    let header_path = PathBuf::from("../cpp/src/tket1-passes.h");
    let library_search_paths = if is_macos() {
        vec!["/opt/homebrew/lib"]
    } else {
        vec![]
    };

    // Check the required files can be found
    assert!(
        lib_path.exists(),
        "{} not found. Ensure the C++ library has been compiled first. \
         If using a non-default library path, you may set the environment \
         variable TKET_LIB_PATH.",
        libtket1_passes_name()
    );
    assert!(header_path.exists(), "tket1-passes.h not found.");

    let lib_dir = path::absolute(&lib_path)
        .unwrap()
        .parent()
        .unwrap()
        .to_owned();

    // Configure when to rerun the build script
    println!("cargo:rerun-if-changed={}", lib_path.display());
    println!("cargo:rerun-if-changed={}", header_path.display());

    // Configure where to find the dynamic libraries (tket1-passes and GMP).
    println!("cargo:rustc-link-search=native={}", lib_dir.display());
    for path in library_search_paths {
        println!("cargo:rustc-link-search=native={path}");
    }

    // Libraries to link in dynamically.
    println!("cargo:rustc-link-lib=tket1-passes");
    if is_macos() {
        println!("cargo:rustc-link-lib=c++");
    } else {
        println!("cargo:rustc-link-lib=stdc++");
    }

    // Configure the linker to find the dynamic libraries at runtime.
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}", lib_dir.display());

    // Generate bindings using bindgen
    let bindings = bindgen::Builder::default()
        .header(header_path.to_str().unwrap())
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .generate()
        .expect("Unable to generate bindings");

    // Write the bindings to the $OUT_DIR/bindings.rs file.
    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap()).join("bindings.rs");
    bindings
        .write_to_file(&out_path)
        .expect("Couldn't write bindings!");
}
