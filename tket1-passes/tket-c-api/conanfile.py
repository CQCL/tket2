from conan import ConanFile
from conan.tools.cmake import CMakeToolchain, CMake, cmake_layout, CMakeDeps


class TketCAPIRecipe(ConanFile):
    name = "tket-c-api"
    version = "0.0.1"
    package_type = "library"

    # Optional metadata
    license = "Apache 2"
    author = "Luca Mondada <luca.mondada@quantinuum.com>"
    description = "C interface to a small subset of TKET1's functionality"
    topics = ("quantum", "computation", "compiler", "c-interface")

    # Binary configuration
    settings = "os", "compiler", "build_type", "arch"
    options = {"shared": [True, False], "fPIC": [True, False]}
    default_options = {"shared": False, "fPIC": True}

    # Sources are located in the same place as this recipe, copy them to the recipe
    exports_sources = "CMakeLists.txt", "src/*", "include/*"

    def config_options(self):
        if self.settings.os == "Windows":
            self.options.rm_safe("fPIC")

    def configure(self):
        if self.options.shared:
            self.options.rm_safe("fPIC")

    def layout(self):
        cmake_layout(self)

    def generate(self):
        deps = CMakeDeps(self)
        deps.generate()
        tc = CMakeToolchain(self)
        tc.generate()

    def build(self):
        cmake = CMake(self)
        cmake.configure()
        cmake.build()

    def package(self):
        cmake = CMake(self)
        cmake.install()

    def package_info(self):
        self.cpp_info.libs = ["tket-c-api"]

    def requirements(self):
        self.requires("tket/2.1.28@tket/stable")


    

