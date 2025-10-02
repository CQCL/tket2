def compile_to_bitcode(pkg_bytes: bytes, opt_level: int = 2) -> bytes:
    """Compile serialized HUGR to LLVM IR bitcode"""
    ...

def compile_to_llvm_ir(pkg_bytes: bytes, opt_level: int = 2) -> str:
    """Compile serialized HUGR to LLVM IR string"""
    ...

def check_hugr(pkg_bytes: bytes) -> None:
    """Load serialized HUGR and validate it.

    Raises:
        HugrReadError if the HUGR is invalid.
    """
    ...

class HugrReadError(Exception):
    """Raised when reading HUGR fails"""

    ...
