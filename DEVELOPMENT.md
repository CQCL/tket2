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

- Rust 1.75+: https://www.rust-lang.org/tools/install

- Poetry: https://python-poetry.org/


Simply run `poetry shell` to activate an environment with all the required dependencies.

## 🏃 Running the tests

The repository root contains a Justfile with the most common development tasks.
Run `just` to see a list.

To manually compile and test the rust code, run:

```bash
cargo test
```

Run the benchmarks with:

```bash
cargo bench
```

Finally, if you have rust nightly installed, you can run `miri` to detect
undefined behaviour in the code. Note that the _devenv_ shell only has rust
stable available.

```bash
cargo +nightly miri test
```

To run the python tests, run:

```bash
maturin develop
pytest
```

You can use the script in [`.github/pre-commit`](.github/pre-commit) to run the test and formatting required by our CI.
To automatically check that before each commit, install it as a hook with:

```bash
ln -s .github/pre-commit $PWD/.git/hooks/pre-commit
# Or, to check before pushing instead
ln -s .github/pre-commit $PWD/.git/hooks/pre-push
```

## 💅 Coding Style

The rustfmt tool is used to enforce a consistent rust coding style. The CI will fail if the code is not formatted correctly. Python code is formatted with black.

To format your code, run:

```bash
# Format rust code
cargo fmt
# Format python code
ruff format .
```

We also check for clippy warnings, which are a set of linting rules for rust. To run clippy, run:

```bash
cargo clippy --all-targets
```

## 📈 Code Coverage

We run coverage checks on the CI. Once you submit a PR, you can review the
line-by-line coverage report on
[codecov](https://app.codecov.io/gh/CQCL/tket2/commits?branch=All%20branches).

To run the rust coverage checks locally, install `cargo-llvm-cov`, generate the report with:
```bash
cargo llvm-cov --lcov > lcov.info
```
and open it with your favourite coverage viewer. In VSCode, you can use
[`coverage-gutters`](https://marketplace.visualstudio.com/items?itemName=ryanluker.vscode-coverage-gutters).

Similarly, to run the python coverage checks locally, install `pytest-cov` and run:
```bash
pytest --cov=./ --cov-report=xml
```

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
