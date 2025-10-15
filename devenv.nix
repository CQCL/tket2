{ pkgs, lib, inputs, ... }:
let
  pkgs-stable = import inputs.nixpkgs-2505 { system = pkgs.stdenv.system; };
  llvmVersion = "14";
  llvmPackages = pkgs-stable."llvmPackages_${llvmVersion}";
in
{
  # https://devenv.sh/packages/
  # on macos frameworks have to be explicitly specified
  # otherwise a linker error occurs on rust packages
  packages = [
    pkgs.just
    pkgs.cargo-insta
    pkgs.cargo-nextest

    # These are required to be able to link to llvm.
    pkgs.libffi
    pkgs.libxml2
    pkgs.zlib
    pkgs.ncurses
  ]
  ++ lib.optionals pkgs.stdenv.isLinux [
    pkgs.stdenv.cc.cc.lib
  ];


  enterShell = ''
    cargo --version
    python --version
    uv --version
  '';

  env = {
    "LLVM_SYS_${llvmVersion}0_PREFIX" = "${llvmPackages.libllvm.dev}";
  };

  # https://devenv.sh/languages/

  languages.rust = {
    enable = true;
    channel = "stable";
    components = [ "rustc" "cargo" "clippy" "rustfmt" "rust-analyzer" ];
  };

  languages.python = {
    enable = true;
    uv = {
      enable = true;
      sync.enable = true;
    };
    venv.enable = true;
  };


}
