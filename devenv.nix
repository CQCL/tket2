{ pkgs, ... }:

{

  # https://devenv.sh/packages/
  packages = with pkgs; [ cargo rustc rust-analyzer rustfmt clippy maturin ];

  # Certain Rust tools won't work without this
  # This can also be fixed by using oxalica/rust-overlay and specifying the rust-src extension
  # See https://discourse.nixos.org/t/rust-src-not-found-and-other-misadventures-of-developing-rust-on-nixos/11570/3?u=samuela. for more details.
  env.RUST_SRC_PATH = "${pkgs.rust.packages.stable.rustPlatform.rustLibSrc}";
  # https://devenv.sh/scripts/
  scripts.hello.exec = "echo Welcome to tket2proto devenv!";

  enterShell = ''
    hello
    cargo --version
  '';

  # https://devenv.sh/languages/
  languages.python.enable = true;
  languages.python.venv.enable = true;
  # https://devenv.sh/pre-commit-hooks/
  # pre-commit.hooks.shellcheck.enable = true;

  # https://devenv.sh/processes/
  # processes.ping.exec = "ping example.com";

  # See full reference at https://devenv.sh/reference/options/
}
