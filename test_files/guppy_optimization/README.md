# Guppy-generated programs we aim to optimize

This directory contains guppy program definitions that we should be able to optimize to a similar level as passing
the equivalent flat program to `pytket`.

Each example is in its own directory, containing:
- A `.py` [uv script](https://docs.astral.sh/uv/guides/scripts/) that defines the guppy program for a pinned version of guppylang.
- If possible, a `.flat.py` script that defines the same program using `comptime` or manual loop peeling.
- A `.opt.py` script that defines the optimized version of the program, which we
  expect to obtain after optimizing the previous versions.

- The `.hugr` files generated for each of the previous scripts.

If the the directory contains a `.generate_mmd` empty file, it will also contain
the `.mmd` mermaid files generated for each of the previous programs. Go to
<http://mermaid.live> to view them.
We do not generate these for large programs, as they get too large to view.

Run `just regenerate` in this directory to regenerate the `.hugr` files.
The guppylang version used is defined in the `justfile`.
The mermaid diagrams will only be generated if the `hugr` CLI is installed.