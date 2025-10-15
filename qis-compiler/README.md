# The HUGR-QIS compiler for Selene

This package provides compilation from HUGR to LLVM adhering
to the Quantinuum QIS.

## Usage

Guppy can produce a HUGR Envelope with:
```python
from guppylang.decorator import guppy

@guppy
def main() -> None:
    # ...

hugr = main.compile()
hugr_envelope = hugr.package.to_bytes()
```

This can then be compiled to LLVM IR or bitcode
using this package:

```python
from qis_compiler import (
    compile_to_llvm_ir,
    compile_to_bitcode,
)

# Compile to LLVM IR for the host system
# and receive it as a string
ir = compile_to_llvm_ir(hugr_envelope)

# Compile to LLVM bitcode for the host system
# and receive it as a bytes object
bitcode = compile_to_bitcode(hugr_envelope)
```

If you wish to target a specific architecture or platform,
you can pass the triple as an argument to the compilation functions:

```python
# Compile to LLVM IR for Apple Silicon
ir_apple_silicon = compile_to_llvm_ir(
    hugr_envelope,
    target_triple="aarch64-apple-darwin",
)
# Compile to LLVM bitcode for x86_64 MSVC
bitcode_apple_silicon = compile_to_bitcode(
    hugr_envelope,
    target_triple="x86_64-windows-msvc",
)
```
