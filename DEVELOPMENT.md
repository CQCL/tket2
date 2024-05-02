# Welcome to the tket2 development guide <!-- omit in toc -->

This guide is intended to help you get started with developing tket2.

If you find any errors or omissions in this document, please [open an issue](https://github.com/CQCL-DEV/tket2/issues/new)!

## #️⃣ Setting up the development environment

You can setup the development environment in two ways:

### The Nix way

The easiest way to setup the development environment is to use the provided
[`devenv.nix`](devenv.nix) file. This will setup a development shell with all the
required dependencies.

To use this, you will need to install [devenv](https://devenv.sh/getting-started/).
Once you have it running, open a shell with:

```bash
devenv shell
```

All the required dependencies should be available. You can automate loading the
shell by setting up [direnv](https://devenv.sh/automatic-shell-activation/).

### Poetry setup

To setup the environment manually you will need:

- Just: https://just.systems/
- Rust `>=1.75`: https://www.rust-lang.org/tools/install
- Poetry `>=1.8`: https://python-poetry.org/

Once you have these installed, install the required python dependencies and setup pre-commit hooks with:

```bash
just setup
```

## 🚀 Local development using the tket2 python library

If you want to use the `tket2` python library directly from the repository, you can build it with:

```bash
just build
```

This will build the python wheels and make them available in the `target/wheels` folder.

Alternatively, you can build the package directly into a virtual environment as an editable package.
That way, you can make changes to the python code and see the changes reflected in your environment.
For this you must have `maturin` installed (you can install it with `pip install maturin`) and run:

```bash
maturin develop
```

## 🏃 Running the tests

To compile and test the code, run:

```bash
just test
# or, to test only the rust code or the python code
just test rust
just test python
```

Run `just` to see all available commands.

## 💅 Coding Style

We use `rustfmt` and `ruff` to enforce a consistent coding style. The CI will fail if the code is not formatted correctly.

To format your code, run:

```bash
just format
```

We also check for linting errors with `clippy`, `ruff` and `mypy`.
To check all linting and formatting, run:

```bash
just check
```

some errors can be fixed automatically with

```bash
just fix
```

## 📈 Code Coverage

We run coverage checks on the CI. Once you submit a PR, you can review the
line-by-line coverage report on
[codecov](https://app.codecov.io/gh/CQCL/tket2/commits?branch=All%20branches).

To run the coverage checks locally, install `cargo-llvm-cov` by running `cargo install cargo-llvm-cov` and then run:
```bash
just coverage
```
This will generate a coverage file that can be opened with your favourite coverage viewer. In VSCode, you can use
[`coverage-gutters`](https://marketplace.visualstudio.com/items?itemName=ryanluker.vscode-coverage-gutters).

## 🌐 Contributing to tket2

We welcome contributions to tket2! Please open [an issue](https://github.com/CQCL/tket2/issues/new) or [pull request](https://github.com/CQCL/tket2/compare) if you have any questions or suggestions.

PRs should be made against the `main` branch, and should pass all CI checks before being merged. This includes using the [conventional commits](https://www.conventionalcommits.org/en/v1.0.0/) format for the PR title.

The general format of a contribution title should be:

```
<type>(<scope>)!: <description>
```

Where the scope is optional, and the `!` is only included if this is a semver breaking change that requires a major version bump.

We accept the following contribution types:

- feat: New features.
- fix: Bug fixes.
- docs: Improvements to the documentation.
- style: Formatting, missing semi colons, etc; no code change.
- refactor: Refactoring code without changing behaviour.
- perf: Code refactoring focused on improving performance.
- test: Adding missing tests, refactoring tests; no production code change.
- ci: CI related changes. These changes are not published in the changelog.
- chore: Updating build tasks, package manager configs, etc. These changes are not published in the changelog.
- revert: Reverting previous commits.
