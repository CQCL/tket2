#!/bin/bash

# Build script for tket1-passes using Conan
set -e

echo "Building tket1-passes..."

# Clean build directory
rm -rf build
mkdir -p build
cd build

# Install dependencies and generate build files with Conan
conan install .. --output-folder=. --build=missing

# Find the toolchain file in the generated directory
TOOLCHAIN_FILE=$(find . -name "conan_toolchain.cmake" | head -1)
if [ -z "$TOOLCHAIN_FILE" ]; then
    echo "Error: Could not find conan_toolchain.cmake"
    exit 1
fi

echo "Using toolchain file: $TOOLCHAIN_FILE"

# Configure with CMake using Conan's generated files
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_TOOLCHAIN_FILE="$TOOLCHAIN_FILE" 

# Build the shared library
make -j$(nproc 2>/dev/null || echo 4)

# Copy the library for Rust usage
make copy_to_lib_dir

echo "Build complete!"

cd ..
echo "Dynamic library available at: $(pwd)/lib/libtket1-passes.dylib"
