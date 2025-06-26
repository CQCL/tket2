cmake_minimum_required(VERSION 3.23)
project(tket1-passes C CXX)

# Enable compile_commands.json generation for VSCode IntelliSense
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

# Set the library properties for shared library creation
set_target_properties(tket1-passes PROPERTIES
    VERSION 1.0.0
    PUBLIC_HEADER "src/tket1-passes.h"
)

# Installation
include(GNUInstallDirs)
install(TARGETS tket1-passes
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

# Custom target to copy the library to a new lib location
add_custom_target(copy_to_lib_dir
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_SOURCE_DIR}/lib
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:tket1-passes> ${CMAKE_CURRENT_SOURCE_DIR}/lib/
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_LINKER_FILE:tket1-passes> ${CMAKE_CURRENT_SOURCE_DIR}/lib/
    DEPENDS tket1-passes
    COMMENT "Copying library and symlinks to lib directory"
)
