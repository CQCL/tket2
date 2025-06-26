from conan import ConanFile  # type: ignore
from conan.tools.cmake import CMakeToolchain, CMake, cmake_layout, CMakeDeps  # type: ignore


class Tket1PassesConan(ConanFile):
    name = "tket1-passes"
    version = "1.0.0"
    package_type = "library"
    license = "Apache 2"
    homepage = "https://github.com/CQCL/tket"
    description = "C interface for tket quantum SDK"
    topics = ("quantum", "computation", "compiler", "c-interface")
    settings = "os", "compiler", "build_type", "arch"
    options = {
        "shared": [True, False],
        "fPIC": [True, False],
    }
    default_options = {
        "shared": True,
        "fPIC": True,
    }
    exports_sources = (
        "CMakeLists.txt",
        "src/tket1-passes.cpp",
        "src/tket1-passes.h",
        "build.sh",
        "README.md",
    )

    def config_options(self):
        if self.settings.os == "Windows":
            del self.options.fPIC

    def configure(self):
        # Only need header libraries from boost
        self.options["boost"].header_only = True

    def layout(self):
        cmake_layout(self)

    def generate(self):
        deps = CMakeDeps(self)
        deps.generate()
        tc = CMakeToolchain(self)
        tc.generate()

    def requirements(self):
        # Use the local tket package
        self.requires("tket/2.1.27@tket/stable")

    def build(self):
        cmake = CMake(self)
        cmake.configure()
        cmake.build()

    def package(self):
        cmake = CMake(self)
        cmake.install()

    def package_info(self):
        self.cpp_info.libs = ["tket1-passes"]
