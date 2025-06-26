//! Build script for the tket1-passes crate
//!
//! Use bindgen to generate Rust bindings for the C library.

use std::env;
use std::path;
use std::path::PathBuf;

fn main() {
    let lib_path = PathBuf::from("../cpp/lib/libtket1-passes.dylib");
    let header_path = PathBuf::from("../cpp/src/tket1-passes.h");

    // Check the required C++ files can be found
    assert!(
        lib_path.exists(),
        "libtket1-passes.dylib not found. Ensure the C++ library has been compiled first."
    );
    assert!(header_path.exists(), "tket1-passes.h not found.");

    let lib_dir = path::absolute(&lib_path)
        .unwrap()
        .parent()
        .unwrap()
        .to_owned();

    // Libraries to link in dynamically.
    println!("cargo:rustc-link-lib=dylib=tket1-passes");
    println!("cargo:rustc-link-lib=c++");
    println!("cargo:rustc-link-lib=gmp");

    // Configure where to find the dynamic libraries (tket1-passes and GMP).
    println!("cargo:rustc-link-search=native={}", lib_dir.display());
    println!("cargo:rustc-link-search=native=/opt/homebrew/opt/gmp/lib");

    // Configure when to rerun the build script
    println!("cargo:rerun-if-changed={}", lib_path.display());
    println!("cargo:rerun-if-changed={}", header_path.display());

    // Configure the linker to find the dynamic libraries at runtime.
    if cfg!(target_os = "macos") {
        println!("cargo:rustc-link-arg=-Wl,-rpath,{}", lib_dir.display());
    }

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
