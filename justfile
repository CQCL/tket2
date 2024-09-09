# List the available commands
help:
    @just --list --justfile {{justfile()}}

# Prepare the environment for development, installing all the dependencies and
# setting up the pre-commit hooks.
setup:
    uv sync
    [[ -n "${TKET2_JUST_INHIBIT_GIT_HOOKS:-}" ]] || uv run pre-commit install -t pre-commit

# Run the pre-commit checks.
check:
    uv run pre-commit run --all-files

# Compile the wheels for the python package.
build:
    cd tket2-py && uv run maturin build --release

# Run all the tests.
test language="[rust|python]" : (_run_lang language \
        "uv run cargo test --all-features --workspace" \
        "uv run maturin develop && uv run pytest"
    )

# Auto-fix all clippy warnings.
fix language="[rust|python]": (_run_lang language \
        "uv run cargo clippy --all-targets --all-features --workspace --fix --allow-staged --allow-dirty" \
        "uv run ruff check --fix"
    )

# Format the code.
format language="[rust|python]": (_run_lang language \
        "uv run cargo fmt" \
        "uv run ruff format"
    )

# Generate a test coverage report.
coverage language="[rust|python]": (_run_lang language \
        "uv run cargo llvm-cov --lcov > lcov.info" \
        "uv run maturin develop && uv run pytest --cov=./ --cov-report=html"
    )

# Runs `compile-rewriter` on the ECCs in `test_files/eccs`
recompile-eccs:
    scripts/compile-test-eccs.sh

# Generate serialized declarations for the tket2 extensions
gen-extensions:
    cargo run -p tket2-hseries gen-extensions -o tket2-py/tket2/extensions/_json_defs

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
