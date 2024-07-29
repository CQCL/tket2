# List the available commands
help:
    @just --list --justfile {{justfile()}}

# Prepare the environment for development, installing all the dependencies and
# setting up the pre-commit hooks.
setup:
    poetry install
    poetry run -- pre-commit install -t pre-commit

# Run the pre-commit checks.
check:
    poetry run -- pre-commit run --all-files

# Compile the wheels for the python package.
build:
    poetry run -- maturin build --release

# Run all the tests.
test language="[rust|python]" : (_run_lang language \
        "poetry run cargo test --all-features --workspace" \
        "poetry run maturin develop && poetry run pytest"
    )

# Auto-fix all clippy warnings.
fix language="[rust|python]": (_run_lang language \
        "poetry run -- cargo clippy --all-targets --all-features --workspace --fix --allow-staged --allow-dirty" \
        "poetry run -- ruff check --fix"
    )

# Format the code.
format language="[rust|python]": (_run_lang language \
        "poetry run cargo fmt" \
        "poetry run ruff format"
    )

# Generate a test coverage report.
coverage language="[rust|python]": (_run_lang language \
        "poetry run -- cargo llvm-cov --lcov > lcov.info" \
        "poetry run -- maturin develop && poetry run pytest --cov=./ --cov-report=html"
    )

# Load a shell with all the dependencies installed
shell:
    poetry shell

# Runs `compile-rewriter` on the ECCs in `test_files/eccs`
recompile-eccs:
    scripts/compile-test-eccs.sh


# Runs a rust and a python command, depending on the `language` variable.
#
# If `language` is set to `rust` or `python`, only run the command for that language.
# Otherwise, run both commands.
_run_lang language rust_cmd python_cmd:
    #!/usr/bin/env bash
    set -euo pipefail
    if [ "{{ language }}" = "rust" ]; then
        set -x
        {{ rust_cmd }}
    elif [ "{{ language }}" = "python" ]; then
        set -x
        {{ python_cmd }}
    else
        set -x
        {{ rust_cmd }}
        {{ python_cmd }}
    fi
