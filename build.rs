#[cfg(feature = "quartz-bench")]
use std::path::PathBuf;

#[cfg(feature = "quartz-bench")]
fn main() {
    let quartz_folder = PathBuf::from("benches/quartz");
    let header = quartz_folder.join("quartz_pattern_match.hpp");
    // Tell cargo to look for shared libraries in the specified directory
    println!(
        "cargo:rustc-link-search={}",
        quartz_folder.join("lib").to_str().unwrap()
    );

    // Tell cargo to tell rustc to link the system bzip2
    // shared library.
    println!("cargo:rustc-link-lib=quartz_runtime");
    println!("cargo:rustc-link-lib=quartz_pattern_match");

    // Tell cargo to invalidate the built crate whenever the wrapper changes
    println!("cargo:rerun-if-changed={}", header.to_str().unwrap());

    // The bindgen::Builder is the main entry point
    // to bindgen, and lets you build up options for
    // the resulting bindings.
    let bindings = bindgen::Builder::default()
        // The input header we would like to generate
        // bindings for.
        .header(header.to_str().unwrap())
        // Tell cargo to invalidate the built crate whenever any of the
        // included header files changed.
        .parse_callbacks(Box::new(bindgen::CargoCallbacks))
        // Finish the builder and generate the bindings.
        .generate()
        // Unwrap the Result and panic on failure.
        .expect("Unable to generate bindings");

    // Write the bindings to the $OUT_DIR/bindings.rs file.
    let out_path = quartz_folder.join("bindings.rs");
    bindings
        .write_to_file(out_path)
        .expect("Couldn't write bindings!");
}

#[cfg(not(feature = "quartz-bench"))]
fn main() {}
