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

### Manual setup

To setup the environment manually you will need:

- Rust: https://www.rust-lang.org/tools/install

- Python 3.10+: https://www.python.org/downloads/

    It is advisable to use a virtual environment for keeping the python
    environment isolated, see
    [`venv`](https://docs.python.org/3/tutorial/venv.html) for more details.

    Install the python development dependencies with:

    ```bash
    pip install -r pyrs/dev-requirements.txt
    ```

If you are testing the `tkcxx` feature to bind to TKET-1, you will need to setup
the `tket-rs` bindings. This requires configuring `conan` for the project, see
the [TKET-rs readme](https://github.com/CQCL-DEV/tket-rs#readme) for more
details.

You can use the git hook in [`.github/pre-commit`](.github/pre-commit) to automatically run the test and check formatting before committing.
To install it, run:

```bash
ln -s .github/pre-commit .git/hooks/pre-commit
# Or, to check before pushing instead
ln -s .github/pre-commit .git/hooks/pre-push
```

## 🏃 Running the tests

To compile and test the rust code, run:

```bash
cargo build
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
cd pyrs
maturin develop
pytest
```

## 💅 Coding Style

The rustfmt tool is used to enforce a consistent rust coding style. The CI will fail if the code is not formatted correctly. Python code is formatted with black.

To format your code, run:

```bash
# Format rust code
cargo fmt
# Format python code
black .
```

We also check for clippy warnings, which are a set of linting rules for rust. To run clippy, run:

```bash
cargo clippy --all-targets
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