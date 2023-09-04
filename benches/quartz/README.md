## Compiling the quartz binaries

The Quartz benchmarking depends on two libraries, `quartz_runtime` and `quart_pattern_match`.

#### `quartz_runtime`
This is obtained by compiling the Quartz project directly from [Github](https://github.com/quantum-compiler/quartz.git).

- `lib/libquartz_runtime.dylib` was compiled on MacOS 12 (x86) from the master branch, commit `0c4afdb` (latest at the time of writing).
- `lib/libquartz_runtime.so` was compiled on linux (x86) from the master branch, commit `0c4afdb` (latest at the time of writing), using `clang 16.0.6`.

#### `quartz_pattern_match`
A library obtained by compiling the `quartz_pattern_match.cpp` file in this folder, using
```
clang++ -O3 -shared -fPIC -o lib/libquartz_pattern_match.dylib --std=c++17 quartz_pattern_match.cpp -IQUARTZ_SRC -Llib -lquartz_runtime
```
where `QUARTZ_SRC` must be replaced by the path to the `src` directory
of the `Quartz` repo on your machine.

- `lib/libquartz_pattern_match.dylib` was compiled on MacOS 12 (x86).
- `lib/libquartz_pattern_match.so` was compiled on linux (x86), using `clang 16.0.6`.