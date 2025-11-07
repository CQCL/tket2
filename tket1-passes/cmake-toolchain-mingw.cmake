# CMake toolchain file for mingw-w64 cross-compilation from Linux
# This file is used in CI builds to ensure CMake uses GCC-style flags instead of MSVC flags
set(CMAKE_SYSTEM_NAME Windows)
set(CMAKE_SYSTEM_PROCESSOR x86_64)

# Specify the cross compiler
set(CMAKE_C_COMPILER x86_64-w64-mingw32-gcc)
set(CMAKE_CXX_COMPILER x86_64-w64-mingw32-g++)

# Tell CMake to use GCC-style flags, not MSVC
set(CMAKE_C_COMPILER_ID "GNU")
set(CMAKE_CXX_COMPILER_ID "GNU")

# Disable MSVC-specific flags
set(CMAKE_C_FLAGS_INIT "")
set(CMAKE_CXX_FLAGS_INIT "")
set(CMAKE_C_FLAGS_RELEASE_INIT "")
set(CMAKE_CXX_FLAGS_RELEASE_INIT "")

# Use Unix Makefiles generator
set(CMAKE_GENERATOR "Unix Makefiles" CACHE STRING "" FORCE)
