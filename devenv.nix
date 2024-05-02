{ pkgs, lib, config, inputs, ... }:
let
  pkgs-2305 = import inputs.nixpkgs-2305 { system = pkgs.stdenv.system; };
  pkgs-poetry = import inputs.poetry-fix { system = pkgs.stdenv.system; };
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
    poetry --version
    export LLVM_COV="${pkgs.llvmPackages_16.libllvm}/bin/llvm-cov"
    export LLVM_PROFDATA="${pkgs.llvmPackages_16.libllvm}/bin/llvm-profdata"
  '';

  # https://devenv.sh/languages/

  languages.rust = {
    enable = true;
    channel = "stable";
    components = [ "rustc" "cargo" "clippy" "rustfmt" "rust-analyzer" ];
  };

  languages.python = {
    enable = true;
    poetry = {
      enable = true;
      activate.enable = true;
      # contains fix to poetry package on macos
      package = pkgs-poetry.poetry;
    };
  };

  # https://devenv.sh/pre-commit-hooks/
  pre-commit.hooks.clippy.enable = true;
  pre-commit.tools.clippy = lib.mkForce config.languages.rust.toolchain.clippy;
  pre-commit.hooks.rustfmt.enable = true;
  pre-commit.tools.rustfmt = lib.mkForce config.languages.rust.toolchain.rustfmt;
}
