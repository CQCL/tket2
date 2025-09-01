# Guppy-generated programs we aim to optimize

This directory contains guppy program definitions that we should be able to optimize to a similar level as passing
the equivalent flat program to `pytket`.

Each example is in its own directory, containing:
- A `.py` [uv script](https://docs.astral.sh/uv/guides/scripts/) that defines the guppy program for a pinned version of guppylang.
- A `.flat.py` script that defines the same program using `comptime` or manual loop peeling, if applicable.
- A `.opt.py` script that defines a hand-optimized version of the program, indicative of the greatest optimization we believe can be achieved by the compiler.

- The `.hugr` files generated for each of the previous scripts.

Run `just regenerate` in this directory to regenerate the `.hugr` files.
The guppylang version used is defined by each script.

If a directory contains a `.generate_mmd` empty file, running
`just mermaid` will generate `.mmd` mermaid files for each of the previous
programs. Go to <http://mermaid.live> to view them. We do not generate these for
large programs, as they get too large to view.
