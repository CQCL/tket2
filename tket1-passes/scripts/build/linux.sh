#!/bin/bash
set -eou pipefail

# Build script for linux targets, designed specifically for
# use within manylinux_2_28 and musllinux_1_2 containers on
# x86_64 and aarch64 architectures.
#
# Note that different container images may have access to
# different tooling, package managers, etc. We assume
# pipx is available. jq is installed from github releases
# if not already present.

output_dir="$1"
if [ -z "${output_dir}" ]; then
  echo "Usage: $0 <output_dir>"
  exit 1
fi;

if ! command -v jq &> /dev/null; then
    # if uname -m is x86_64, download arm64 version of jq and use the corresponding hash
    # if uname -m is aarch64, download aarch64 version of jq and use the corresponding hash
    case "$(uname -m)" in
        "aarch64")
            jq_filename="jq-linux-arm64";
            jq_sha256="6bc62f25981328edd3cfcfe6fe51b073f2d7e7710d7ef7fcdac28d4e384fc3d4";
        ;;
        "x86_64")
            jq_filename="jq-linux-amd64";
            jq_sha256="020468de7539ce70ef1bceaf7cde2e8c4f2ca6c3afb84642aabc5c97d9fc2a0d"
        ;;
        *)
            echo "Unsupported architecture: $(uname -m)"
            exit 1
        ;;
    esac
    echo "jq not found. Downloading..."
    curl -LO https://github.com/jqlang/jq/releases/download/jq-1.8.1/${jq_filename}
    echo "Verifying jq checksum..."
    echo "${jq_sha256}  ${jq_filename}" | sha256sum -c
    chmod +x ${jq_filename}
    echo "Installing jq..."
    mv ${jq_filename} /usr/bin/jq
    echo "jq installed successfully."
fi

# Install and run conan
pipx install conan
conan profile detect
conan remote add tket-libs https://quantinuumsw.jfrog.io/artifactory/api/conan/tket1-libs --index 0
conan install /tket1-passes \
  --build=missing \
  --options="tket-c-api/*:shared=True" \
  --format=json \
  --output-folder=/tmp/conan_output \
  --out-file=/tmp/conanbuild.json

# Extract the library folder path from the conan JSON output, and
# copy the built artifacts to the host tket1-passes/out directory
lib_folder=$(jq -r '.graph.nodes."1".package_folder' /tmp/conanbuild.json)
echo "Installation folder: ${lib_folder}"
chmod -R a+r "${lib_folder}"
echo "Copying built artifacts to output directory: ${output_dir}"
cp -r "${lib_folder}"/* "${output_dir}"
