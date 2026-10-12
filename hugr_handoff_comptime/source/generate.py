"""Generate the same handoff constructions with static circuit-building loops."""

import ast
import hashlib
import json
import sys
import types
from pathlib import Path

from source_macros import PythonBody, SourceMacros

HERE = Path(__file__).resolve().parent.parent
RUNTIME_SOURCE = HERE.parent / "hugr_handoff_runtime/source/generate.py"


class TraceEntrypoints(ast.NodeTransformer):
    """Trace the entrypoints and execute their local helpers as Python."""

    def visit_FunctionDef(self, node):
        if any(
            isinstance(d, ast.Name) and d.id == "guppy" for d in node.decorator_list
        ):
            if any(isinstance(n, ast.With) for n in ast.walk(node)):
                return node
            node = PythonBody().visit(node)
            if node.name in {"main", "adder_main", "chain_main"}:
                decorator = ast.parse("guppy.comptime", mode="eval").body
                for child in ast.walk(decorator):
                    ast.copy_location(child, node)
                node.decorator_list = [decorator]
                node.returns = ast.Constant(None)
            return node
        return self.generic_visit(node)


def main():
    tree = TraceEntrypoints().visit(ast.parse(RUNTIME_SOURCE.read_text()))
    ast.fix_missing_locations(tree)
    module = types.ModuleType("handoff_constructions")
    sys.modules[module.__name__] = module
    namespace = module.__dict__
    namespace["__file__"] = str(RUNTIME_SOURCE)
    exec(compile(tree, str(RUNTIME_SOURCE), "exec"), namespace)
    macros = SourceMacros()
    macros.prepare_runtime_helpers()
    namespace.update({name: macros.adapt(value) for name, value in namespace.items()})
    namespace.update(
        HERE=HERE,
        ARTIFACT_DIR=HERE / "artifacts",
        MANIFEST_PATH=HERE / "manifest.json",
    )
    namespace["generate"]()
    (HERE / "source_audit.json").write_text(
        json.dumps(macros.sources, indent=2, sort_keys=True) + "\n"
    )
    manifest = json.loads((HERE / "manifest.json").read_text())
    manifest["generation"] = (
        "same library bodies executed during @guppy.comptime tracing"
    )
    manifest["runtime_source_sha256"] = hashlib.sha256(
        RUNTIME_SOURCE.read_bytes()
    ).hexdigest()
    manifest["comptime_expansion_added"] = True
    (HERE / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
