#!/usr/bin/env bash

# This script calls `compile-rewriter` to recompile all ECC sets in the
# `test_files/eccs/` directory.
#
# Note that the output is not deterministic, so the resulting `.rwr` will always
# be different even if the input and the compiler are the same.
set -euo pipefail

# Get the directory of this script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ECCS_DIR="$DIR/../test_files/eccs"

# Command to call compile-rewriter with cargo
COMPILE_REWRITER="cargo run --release -p compile-rewriter --"

# Compile all json files into `.rwr`
for ecc in $ECCS_DIR/*.json; do
    echo "-------- Compiling $ecc"
    $COMPILE_REWRITER -i "$ecc" -o "${ecc%.json}.rwr"
done

# Additional hard-coded step:
# The python package contains a copy of the nam_6_3 and cliffordt_5_3 eccs,
# which must be manually copied.
PY_ECCS_DIR="$DIR/../tket-eccs/src/tket_eccs/data"
for file in "nam_6_3.rwr" "clifford_t_6_3.rwr"; do
    source_file="$ECCS_DIR/$file"
    dest_file="$PY_ECCS_DIR/$file"
    
    echo "Copying $source_file to $dest_file"
    cp -f "$source_file" "$dest_file"
done

echo "Done!"
