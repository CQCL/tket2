{ pkgs, lib, config, inputs, ... }:
let
  pkgs-2305 = import inputs.nixpkgs-2305 { system = pkgs.stdenv.system; };
in
{
  # https://devenv.sh/packages/
  # on macos frameworks have to be explicitly specified
  # otherwise a linker error ocurs on rust packages
  packages = [
    pkgs.just
    pkgs.llvmPackages_16.libllvm
    # cargo-llvm-cov is currently marked broken on nixpkgs unstable
    pkgs-2305.cargo-llvm-cov
  ]
  ++ lib.optionals pkgs.stdenv.isLinux [
    pkgs.stdenv.cc.cc.lib
    pkgs.stdenv.cc.cc
  ]
  ++ lib.optionals pkgs.stdenv.isDarwin (
    with pkgs.darwin.apple_sdk; [
      frameworks.CoreServices
      frameworks.CoreFoundation
    ]
  );

  # Certain Rust tools won't work without this
  # This can also be fixed by using oxalica/rust-overlay and specifying the rust-src extension
  # See https://discourse.nixos.org/t/rust-src-not-found-and-other-misadventures-of-developing-rust-on-nixos/11570/3?u=samuela. for more details.
  #env.RUST_SRC_PATH = "${pkgs.rust.packages.stable.rustPlatform.rustLibSrc}";
  # https://devenv.sh/scripts/
  scripts.hello.exec = "echo Welcome to tket2proto devenv!";

  enterShell = ''
    hello
    cargo --version
    python --version
    uv --version
    export LLVM_COV="${pkgs.llvmPackages_16.libllvm}/bin/llvm-cov"
    export LLVM_PROFDATA="${pkgs.llvmPackages_16.libllvm}/bin/llvm-profdata"
    export TKET2_JUST_INHIBIT_GIT_HOOKS=1
    export LD_LIBRARY_PATH="${config.env.DEVENV_PROFILE}/lib''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}";
    just setup || true
    source .venv/bin/activate
  '';

  env = {
    PYO3_PYTHON = "${config.env.DEVENV_PROFILE}/bin/python";
  };

  languages.rust = {
    enable = true;
    channel = "stable";
    components = [ "rustc" "cargo" "clippy" "rustfmt" "rust-analyzer" ];
  };

  languages.python = {
    enable = true;
    uv = {
      enable = true;
    };
  };

}
