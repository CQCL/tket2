# CMake toolchain file for mingw-w64 cross-compilation from Linux
# This file is used in CI builds to ensure CMake uses GCC-style flags instead of MSVC flags
# Note: This file is included AFTER Conan's toolchain file, so Conan's CMAKE_PREFIX_PATH is already set

set(CMAKE_SYSTEM_NAME Windows)
set(CMAKE_SYSTEM_PROCESSOR x86_64)

# Specify the cross compiler
set(CMAKE_C_COMPILER x86_64-w64-mingw32-gcc CACHE FILEPATH "C compiler" FORCE)
set(CMAKE_CXX_COMPILER x86_64-w64-mingw32-g++ CACHE FILEPATH "C++ compiler" FORCE)

# Tell CMake to use GCC-style flags, not MSVC
set(CMAKE_C_COMPILER_ID "GNU" CACHE STRING "C compiler ID" FORCE)
set(CMAKE_CXX_COMPILER_ID "GNU" CACHE STRING "C++ compiler ID" FORCE)

# Ensure package discovery works by preserving CMAKE_PREFIX_PATH and setting find modes
# Conan sets CMAKE_PREFIX_PATH, so we need to ensure packages can be found
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE BOTH)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE BOTH)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY BOTH)

# Clear MSVC-specific flags that might be set by Conan's toolchain
# We do this after Conan's toolchain has run, so we can filter out MSVC flags
if(CMAKE_CXX_FLAGS MATCHES "/WX|/EHsc")
  string(REGEX REPLACE "/WX[^ ]*" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
  string(REGEX REPLACE "/EHsc[^ ]*" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}" CACHE STRING "C++ flags" FORCE)
endif()
if(CMAKE_C_FLAGS MATCHES "/WX")
  string(REGEX REPLACE "/WX[^ ]*" "" CMAKE_C_FLAGS "${CMAKE_C_FLAGS}")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}" CACHE STRING "C flags" FORCE)
endif()
