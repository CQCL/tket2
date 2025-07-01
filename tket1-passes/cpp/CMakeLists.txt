cmake_minimum_required(VERSION 3.23)
project(tket1-passes C CXX)

# Enable compile_commands.json generation for IDE integration
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_EXTENSIONS OFF)

# Conan will generate the necessary find_package calls
# The tket dependency is handled by Conan
find_package(tket REQUIRED)

# Create the shared library
add_library(tket1-passes SHARED
    src/tket1-passes.cpp
)

# Link against tket and all its dependencies (since tket links them as PRIVATE)
target_link_libraries(tket1-passes PRIVATE tket::tket)

# Installation
include(GNUInstallDirs)

# Set the library destination based on environment variable or default
if(DEFINED ENV{TKET_LIB_PATH})
    if(IS_ABSOLUTE $ENV{TKET_LIB_PATH})
        set(LIB_DESTINATION $ENV{TKET_LIB_PATH})
    else()
        # Path is given relative to the base directory of the repo
        set(LIB_DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/../../$ENV{TKET_LIB_PATH})
    endif()
else()
    # Default path: tket1-passes/lib
    set(LIB_DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/../lib)
endif()

# Install the library to the lib directory in the parent (tket1-passes) directory
install(TARGETS tket1-passes
    LIBRARY DESTINATION ${LIB_DESTINATION}
)

# Create the lib directory if it doesn't exist
install(DIRECTORY DESTINATION ${LIB_DESTINATION})
