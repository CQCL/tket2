//! Build script for the tket1-passes crate
//!
//! Use bindgen to generate Rust bindings for the C library.

use conan2::ConanInstall;
use std::env;
use std::path::PathBuf;

fn is_macos() -> bool {
    let target = env::var("TARGET").unwrap();
    target.contains("apple")
}

fn main() {
    // // Configure when to rerun the build script
    println!("cargo:rerun-if-changed=conanfile.txt");
    println!("cargo:rerun-if-changed=build.rs");

    // Get all dependencies from conan
    let conan_install = ConanInstall::new()
        .build("missing")
        .detect_profile()
        .run()
        .parse();
    conan_install.emit();

    let Some(header_path) = conan_install
        .include_paths()
        .into_iter()
        .map(|incl_path| incl_path.join("tket-c-api.h"))
        .find(|header| header.exists())
    else {
        panic!("required tket-c-api.h header not found");
    };

    // Link in standard C++ library.
    if is_macos() {
        println!("cargo:rustc-link-lib=c++");
    } else {
        println!("cargo:rustc-link-lib=stdc++");
    }

    // Generate bindings using bindgen
    let bindings = bindgen::Builder::default()
        .header(header_path.to_str().unwrap())
        .generate()
        .expect("Unable to generate bindings");

    // Write the bindings to the $OUT_DIR/bindings.rs file.
    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap()).join("bindings.rs");
    bindings
        .write_to_file(&out_path)
        .expect("Couldn't write bindings!");
}
