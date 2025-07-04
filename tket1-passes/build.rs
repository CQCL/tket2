//! Build script for the tket1-passes crate
//!
//! Use bindgen to generate Rust bindings for the C library.

use std::env;
use std::path::{Path, PathBuf};

fn is_macos() -> bool {
    let target = env::var("TARGET").unwrap();
    target.contains("apple")
}

fn is_windows() -> bool {
    let target = env::var("TARGET").unwrap();
    target.contains("windows")
}

fn main() {
    // // Configure when to rerun the build script
    println!("cargo:rerun-if-env-changed=TKET_C_API_PATH");
    println!("cargo:rerun-if-changed=conanfile.txt");
    println!("cargo:rerun-if-changed=build.rs");

    let custom_tket_path = env::var("TKET_C_API_PATH").ok().map(PathBuf::from);

    let header_path = if let Some(path) = custom_tket_path {
        cargo_set_custom_lib_path(&path.join("lib"));
        path.join("include").join("tket-c-api.h")
    } else {
        // Get dependencies from conan
        let conan_install = conan2::ConanInstall::new()
            .build("missing")
            .detect_profile()
            .run()
            .parse();
        conan_install.emit();

        // Get header path
        conan_install
            .include_paths()
            .into_iter()
            .map(|incl_path| incl_path.join("tket-c-api.h"))
            .find(|header| header.exists())
            .expect("required tket-c-api.h header not found")
    };

    assert!(
        header_path.exists(),
        "header not found at {}",
        header_path.display()
    );

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

fn cargo_set_custom_lib_path(search_path: &Path) {
    println!("cargo:rustc-link-search={}", search_path.display());

    let lib_name = if is_windows() {
        "libtket-c-api"
    } else {
        "tket-c-api"
    };
    println!("cargo:rustc-link-lib={lib_name}");
}
