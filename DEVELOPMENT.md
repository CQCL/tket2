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

<!-- TODO: Remove once the pre-compiled binaries are available. -->
Currently you will need to compile the tket C API manually. To do this, run:
```bash
just setup-tket-c-api
```
This step will be removed in the future, once the pre-compiled binaries are available.

### Manual setup

To setup the environment manually you will need:

- Just: <https://just.systems/>
- Rust `>=1.85`: <https://www.rust-lang.org/tools/install>
- uv `>=0.3`: docs.astral.sh/uv/getting-started/installation

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

We also use various linters to catch common mistakes and enforce best practices. To run these, use:

```bash
just check
```

To quickly fix common issues, run:

```bash
just fix
# or, to fix only the rust code or the python code
just fix rust
just fix python
```

## 📈 Code Coverage

We run coverage checks on the CI. Once you submit a PR, you can review the
line-by-line coverage report on
[codecov](https://app.codecov.io/gh/CQCL/tket2/commits?branch=All%20branches).

To run the coverage checks locally, first install `cargo-llvm-cov`.

```bash
cargo install cargo-llvm-cov
```

Then run the tests:

```bash
just coverage
```

This will generate a coverage file that can be opened with your favourite coverage viewer. In VSCode, you can use
[`coverage-gutters`](https://marketplace.visualstudio.com/items?itemName=ryanluker.vscode-coverage-gutters).

## 🌐 Contributing to tket2

We welcome contributions to tket2! Please open [an issue](https://github.com/CQCL/tket2/issues/new) or [pull request](https://github.com/CQCL/tket2/compare) if you have any questions or suggestions.

PRs should be made against the `main` branch, and should pass all CI checks before being merged. This includes using the [conventional commits](https://www.conventionalcommits.org/en/v1.0.0/) format for the PR title.

Some tests may be skipped based on the changes made. To run all the tests in
your PR mark it with a 'run-ci-checks' label and push new commits to it.

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

## :shipit: Releasing new versions

We use automation to bump the version number and generate changelog entries
based on the [conventional commits](https://www.conventionalcommits.org/en/v1.0.0/) labels. Release PRs are created automatically
for each package when new changes are merged into the `main` branch. Once the PR is
approved by someone in the [release team](.github/CODEOWNERS) and is merged, the new package
is published on PyPI or crates.io as appropriate.

The changelog can be manually edited before merging the release PR. Note however
that modifying the diff before other changes are merged will cause the
automation to close the release PR and create a new one to avoid conflicts.

### Rust crate release

Rust releases are managed by `release-plz`. This tool will automatically detect
breaking changes even when they are not marked as such in the commit message,
and bump the version accordingly.

To modify the version being released, update the `Cargo.toml`,
`CHANGELOG.md`, PR name, and PR description in the release PR with the desired version. You may also have to update the dates.
Rust pre-release versions should be formatted as `0.1.0-alpha.1` (or `-beta`, or `-rc`).

### Python package release

Python releases are managed by `release-please`. This tool always bumps the
minor version (or the pre-release version if the previous version was a
pre-release).

To override the version getting released, you must merge a PR to `main` containing
`Release-As: 0.1.0` in the description.
Python pre-release versions should be formatted as `0.1.0a1` (or `b1`, `rc1`).

### Patch releases

Sometimes we need to release a patch version to fix a critical bug, but we don't want
to include all the changes that have been merged into the main branch. In this case,
you can create a new branch from the latest release tag and cherry-pick the commits
you want to include in the patch release.

#### Rust patch releases

You can use [`release-plz`](https://release-plz.ieni.dev/) to automatically generate the changelogs and bump the package versions.

```bash
# If you have cargo-semver-checks installed,
# release-plz will ensure your changes don't break the semver rules.
cargo install cargo-semver-checks --locked
# Analyze the new comments to generate the changelogs / bump the versions
release-plz update
```

Once the branch is ready, create a draft PR so that the release team can review
it.

Now someone from the release team can run `release-plz` on the **unmerged**
branch to create the github releases and publish to crates.io.

```bash
# Make sure you are logged in to `crates.io`
cargo login <your_crates_io_token>
# Get a github token with permissions to create releases
GITHUB_TOKEN=<your_github_token>
# Run release-plz
release-plz release --git-token $GITHUB_TOKEN
```

#### Python patch releases

You will need to modify the version and changelog manually in this case. Check
the existing release PRs for examples on how to do this. Once the branch is
ready, create a draft PR so that the release team can review it.

The wheel building process and publication to PyPI is handled by the CI.
Just create a [github release](https://github.com/CQCL/tket2/releases/new) from the **unmerged** branch.
The release tag should follow the format used in the previous releases, e.g. `tket2-py-v0.1.1`.
