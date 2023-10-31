{ pkgs, lib, config, ... }:

{

  # https://devenv.sh/packages/
  # on macos frameworks have to be explicitly specified 
  # otherwise a linker error ocurs on rust packages
  packages = lib.optionals pkgs.stdenv.isDarwin (with pkgs.darwin.apple_sdk; [
    frameworks.CoreServices
    frameworks.CoreFoundation
    pkgs.just
  ]);

  # Certain Rust tools won't work without this
  # This can also be fixed by using oxalica/rust-overlay and specifying the rust-src extension
  # See https://discourse.nixos.org/t/rust-src-not-found-and-other-misadventures-of-developing-rust-on-nixos/11570/3?u=samuela. for more details.
  #env.RUST_SRC_PATH = "${pkgs.rust.packages.stable.rustPlatform.rustLibSrc}";
  # https://devenv.sh/scripts/
  scripts.hello.exec = "echo Welcome to tket2proto devenv!";

  enterShell = ''
    hello
    cargo --version
  '';

  # https://devenv.sh/languages/

  languages.rust = {
    enable = true;
    components = [ "rustc" "cargo" "clippy" "rustfmt" "rust-analyzer" ];
  };

  languages.python = {
    enable = true;

    venv.enable = true;
    venv.requirements = "-r ${config.env.DEVENV_ROOT}/pyrs/dev-requirements.txt";
  };

  # https://devenv.sh/pre-commit-hooks/
  pre-commit.hooks.clippy.enable = true;
  pre-commit.tools.clippy = lib.mkForce config.languages.rust.toolchain.clippy;
  pre-commit.hooks.rustfmt.enable = true;
  pre-commit.tools.rustfmt = lib.mkForce config.languages.rust.toolchain.rustfmt;
}
