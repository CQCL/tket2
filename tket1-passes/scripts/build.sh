#!/bin/bash
set -eou pipefail
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <target> <compatibility> <output_dir>"
    echo "Examples:"
    echo "   $0 x86_64-unknown-linux-gnu manylinux_2_28_x86_64 ./out"
    echo "   $0 aarch64-apple-darwin 14.0 ./out"
    echo "   $0 x86_64-pc-windows-msvc msvc ./out"
    exit 1
fi

target="$1"
compatibility="$2"
output_dir="$3"

if [ ! -d "${output_dir}" ]; then
    echo "Output directory ${output_dir} does not exist. Creating it."
    mkdir -p "${output_dir}"
fi
output_abs="$(realpath "${output_dir}")"

# For linux builds use docker containers for compatibility
case "${target}" in
    *"linux"*)
        echo "Building tket-c-api in a ${compatibility} container..."
        # We mount the tket1-passes directory as read-only to prevent
        # temporary files from being written to it by conan.
        #
        # The output directory is mounted to /out in the container,
        # and is writable.
        docker run -i \
            -v "$(realpath $(dirname "$0")/../)":/tket1-passes:ro \
            -v "${output_abs}":/out \
            quay.io/pypa/${compatibility}:latest \
            /bin/bash /tket1-passes/scripts/build/linux.sh /out
    ;;
    *"apple-darwin")
        echo "Building tket-c-api with MACOSX_DEPLOYMENT_TARGET=${compatibility}..."
        MACOSX_DEPLOYMENT_TARGET="${compatibility}" \
          bash -c "$(dirname "$0")/build/macos.sh ${output_abs}"
    ;;
    *"windows-msvc")
        echo "Building tket-c-api for Windows with MSVC..." 
        pwsh -File "$(dirname "$0")"/build/windows.ps1 ${output_abs}
    ;;
    *)
      echo "Unsupported target: ${target}"
      exit 1
    ;;
esac
