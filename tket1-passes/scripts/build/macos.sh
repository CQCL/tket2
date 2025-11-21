#!/bin/bash
set -eou pipefail

# Build script for macOS targets. A MACOSX_DEPLOYMENT_TARGET must
# be set in the environment before calling this script.

if [ "$#" -ne 1 ]; then
  echo "Usage: $0 <output_dir>"
  exit 1
fi

output_dir="$1"

echo "Building tket-c-api on aarch64-apple-darwin with MACOSX_DEPLOYMENT_TARGET=${MACOSX_DEPLOYMENT_TARGET}..."

cd "$(dirname "$0")/"../../

brew install jq

# Install and run conan
pipx install conan
conan profile detect
conan remote add tket-libs https://quantinuumsw.jfrog.io/artifactory/api/conan/tket1-libs --index 0
conan install tket1-passes \
  --build=missing \
  --options="tket-c-api/*:shared=True" \
  --format=json \
  --output-folder=/tmp/conan_output \
  --out-file=/tmp/conanbuild.json

# Extract the library folder path from the conan JSON output, and
# copy the built artifacts to tket1-passes/out
lib_folder=$(jq -r '.graph.nodes."1".package_folder' /tmp/conanbuild.json)
cp -r "${lib_folder}"/* "${output_dir}"
