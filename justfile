# List the available commands
help:
    @just --list --justfile {{justfile()}}

# Run all the rust tests
test:
    cargo test --all-features

# Auto-fix all clippy warnings
fix:
    cargo clippy --all-targets --all-features --workspace --fix --allow-staged

# Build the python package wheels
pybuild:
    maturin build --release

# Build the python package for local development
pydevelop:
    maturin develop

# Run the python tests
pytest: pydevelop
    pytest

# Run the pre-commit checks
check:
    ./.github/pre-commit

# Format the code
format:
    cargo fmt
    ruff format .