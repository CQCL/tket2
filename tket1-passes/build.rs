//! Build script for the tket1-passes crate
//!
//! Use bindgen to generate Rust bindings for the C library.

use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

const CONAN_REMOTE: &str = "https://quantinuumsw.jfrog.io/artifactory/api/conan/tket1-libs";

fn main() {
    // // Configure when to rerun the build script
    println!("cargo:rerun-if-env-changed=TKET_C_API_PATH");
    println!("cargo:rerun-if-changed=conanfile.txt");
    println!("cargo:rerun-if-changed=build.rs");

    let custom_tket_path = env::var("TKET_C_API_PATH").ok().map(PathBuf::from);

    let target = SupportedPlatform::from_target_str(&env::var("TARGET").unwrap());

    let header_path = if let Some(path) = custom_tket_path {
        cargo_set_custom_lib_path(&path.join("lib"), target);
        path.join("include").join("tket-c-api.h")
    } else {
        // Get dependencies from conan

        // 1. Add the conan remote to get the tket-c-api source and binaries
        add_conan_remote_if_missing(CONAN_REMOTE);

        // 2. Select one of the pre-built conan profiles if possible (fallback to default)
        let profile_name = target
            .map(|t| t.get_prebuilt_conan_profile())
            .unwrap_or_else(|| {
                println!("cargo:warning=Unrecognized platform. Defaulting to default conan profile and building from source.");
                "default"
            });
        let profile_path = Path::new("./conan-profiles").join(profile_name);
        println!("Using conan profile: {}", profile_path.display());

        // 3. Prepare the conan install command
        let mut conan_install = conan2::ConanInstall::new();
        conan_install
            .build_type("Release")
            .profile(profile_path.to_str().unwrap());

        if profile_name == "default" {
            conan_install.detect_profile().build("missing");
        }

        // 4. Run the conan install command and point cargo to the correct install paths
        let cargo_instructions = conan_install.run().parse();
        cargo_instructions.emit();

        // 5. Get header path for bindgen
        cargo_instructions
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
    if target.is_some_and(|t| t.is_macos()) {
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

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum SupportedPlatform {
    MacOsX86,
    MacOsArm,
    WindowsX86,
    LinusX86,
}

impl SupportedPlatform {
    fn from_target_str(target: &str) -> Option<Self> {
        if target.contains("apple") && target.contains("x86") {
            Some(SupportedPlatform::MacOsX86)
        } else if target.contains("apple") && target.contains("aarch64") {
            Some(SupportedPlatform::MacOsArm)
        } else if target.contains("windows") && target.contains("x86") {
            Some(SupportedPlatform::WindowsX86)
        } else if target.contains("linux") && target.contains("x86") {
            Some(SupportedPlatform::LinusX86)
        } else {
            None
        }
    }

    fn is_macos(&self) -> bool {
        matches!(
            self,
            SupportedPlatform::MacOsX86 | SupportedPlatform::MacOsArm
        )
    }

    fn get_prebuilt_conan_profile(&self) -> &'static str {
        // Corresponds to profiles which have pre-built binaries in the tket-libs conan remote
        // See ./conan-profiles/README.md for more details
        match self {
            SupportedPlatform::MacOsX86 => "macos-13",
            SupportedPlatform::MacOsArm => "macos-15",
            SupportedPlatform::WindowsX86 => "windows-2025",
            SupportedPlatform::LinusX86 => "linux-x86_64-gcc14",
        }
    }
}

fn add_conan_remote_if_missing(conan_remote: &str) {
    let check_exists = Command::new("conan")
        .args(["remote", "list", "-fjson"])
        .output()
        .ok();

    let stdout_str =
        check_exists.map(|output| String::from_utf8_lossy(&output.stdout).into_owned());
    if stdout_str.is_none_or(|s| !s.contains(conan_remote)) {
        println!("Adding conan remote: {conan_remote}");
        Command::new("conan")
            .args(["remote", "add", "tket-libs", conan_remote, "--index", "0"])
            .output()
            .unwrap();
    }
}

fn cargo_set_custom_lib_path(search_path: &Path, target: Option<SupportedPlatform>) {
    println!("cargo:rustc-link-search={}", search_path.display());

    let lib_name = if target.is_some_and(|t| t == SupportedPlatform::WindowsX86) {
        "libtket-c-api"
    } else {
        "tket-c-api"
    };
    println!("cargo:rustc-link-lib={lib_name}");
}
