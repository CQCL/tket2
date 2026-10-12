"""Execute the library's circuit-building bodies during Guppy tracing."""

import ast
import hashlib
import inspect
import textwrap
from dataclasses import make_dataclass
from pathlib import Path

from guppylang.defs import GuppyFunctionDefinition
from guppylang.std.quantum import discard, measure


class StaticArray(list):
    """Keep the library's take/put bookkeeping in Python during tracing."""

    def take(self, index):
        value = self[index]
        assert value is not None
        self[index] = None
        return value

    def put(self, value, index):
        assert self[index] is None
        self[index] = value

    def is_borrowed(self, index):
        return self[index] is None

    def discard_all_taken(self):
        assert all(value is None for value in self)


def static_array(*values):
    if len(values) == 1 and inspect.isgenerator(values[0]):
        return StaticArray(values[0])
    return StaticArray(values)


class StaticOption:
    """Represent the array helpers' compile-time ownership slots."""

    def __init__(self, value=None):
        self.value = value

    @classmethod
    def __class_getitem__(cls, _):
        return cls

    def swap(self, other):
        old = StaticOption(self.value)
        self.value = other.value
        return old

    def unwrap(self):
        assert self.value is not None
        return self.value

    def unwrap_nothing(self):
        assert self.value is None


class Macro:
    """Lazily adapt a source function, including its recursive dependencies."""

    def __init__(self, owner, function):
        self.owner = owner
        self.function = function
        self.compiled = None

    def __getitem__(self, _):
        return self

    def __call__(self, *args, **kwargs):
        if self.compiled is None:
            self.compiled = self.owner.compile(self.function)
        return self.compiled(*args, **kwargs)


class PythonBody(ast.NodeTransformer):
    """Remove Guppy typing/decorators while retaining executable statements."""

    def visit_FunctionDef(self, node):
        dimensions = []
        generic_names = {p.name for p in node.type_params}
        for arg in node.args.posonlyargs + node.args.args + node.args.kwonlyargs:
            annotation = arg.annotation
            if isinstance(annotation, ast.BinOp):
                annotation = annotation.left
            if isinstance(annotation, ast.Subscript):
                if isinstance(annotation.value, ast.Name) and annotation.value.id in {
                    "array",
                    "frozenarray",
                }:
                    items = annotation.slice.elts
                    size = items[1]
                    if isinstance(size, ast.Name) and size.id in generic_names:
                        dimensions.append(
                            ast.parse(f"{size.id} = len({arg.arg})").body[0]
                        )
            arg.annotation = None
        node.type_params = []
        node.decorator_list = []
        node.returns = None
        node = self.generic_visit(node)
        node.body = dimensions + node.body
        return node

    def visit_AnnAssign(self, node):
        if node.value is None:
            return None
        return ast.copy_location(
            ast.Assign(targets=[node.target], value=self.visit(node.value)), node
        )


class SourceMacros:
    """Adapt guppyalgos bodies without replacing their quantum constructions."""

    def __init__(self):
        self.cache = {}
        self.sources = {}

    def adapt(self, value):
        if id(value) in self.cache:
            return self.cache[id(value)]
        raw = getattr(value, "wrapped", None)
        function = getattr(raw, "python_func", value)
        if inspect.isfunction(function) and function.__module__.startswith(
            "guppyalgos"
        ):
            # Leave genuine measurement feed-forward in the original Guppy leaf
            if function.__module__.startswith(
                (
                    "guppyalgos.primitives.gate_decompositions.and_op.and_uncompute",
                    "guppyalgos.primitives.measurement",
                    "guppyalgos.algorithms.state_preparation.multiplexor_prep",
                )
            ):
                return value
            macro = Macro(self, function)
            self.cache[id(value)] = macro
            return macro
        python_class = getattr(raw, "python_class", None)
        if python_class is not None and python_class.__module__.startswith(
            "guppyalgos"
        ):
            fields = list(python_class.__annotations__)
            cls = make_dataclass(python_class.__name__, fields)
            self.cache[id(value)] = cls
            for name, method in vars(python_class).items():
                if isinstance(method, GuppyFunctionDefinition):
                    macro = self.adapt(method)
                    setattr(
                        cls, name, lambda obj, *a, _m=macro, **kw: _m(obj, *a, **kw)
                    )
            return cls
        return value

    def compile(self, function):
        source = textwrap.dedent(inspect.getsource(function))
        tree = ast.parse(source)
        tree = PythonBody().visit(tree)
        ast.fix_missing_locations(tree)
        environment = dict(function.__globals__)
        environment.update(inspect.getclosurevars(function).nonlocals)
        for name in {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}:
            if name in environment:
                environment[name] = self.adapt(environment[name])
        environment.update(
            array=static_array,
            comptime=lambda value: value,
            some=StaticOption,
            nothing=StaticOption,
        )
        # Generic application selects types only, which Python tracing already knows
        function_names = {
            n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
        }
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Subscript):
                base = node.func.value
                if isinstance(base, ast.Name) and base.id in function_names:
                    node.func = base
        ast.fix_missing_locations(tree)
        filename = inspect.getsourcefile(function)
        self.sources[f"{function.__module__}.{function.__qualname__}"] = {
            "file": filename,
            "source": source,
            "adapted": ast.unparse(tree),
            "source_file_sha256": hashlib.sha256(
                Path(filename).read_bytes()
            ).hexdigest(),
        }
        exec(compile(tree, filename, "exec"), environment)
        return environment[function.__name__]

    def prepare_runtime_helpers(self):
        from guppyalgos.utils.guppy import gates

        def transversal(op, *args):
            binary = len(args) >= 2 and isinstance(args[1], list)
            count = 2 if binary else 1
            registers, bounds = args[:count], args[count:]
            if not bounds:
                helper = gates.transversal2_all if binary else gates.transversal1_all
            elif len(bounds) == 2:
                helper = (
                    gates.traversal2_start_end
                    if binary
                    else gates.transversal1_start_end
                )
            else:
                helper = gates.transversal2 if binary else gates.transversal1
            self.adapt(helper)(op, *registers, *bounds)

        self.cache[id(gates.transversal)] = transversal
        self.cache[id(gates.apply_bitstring)] = self.adapt(gates._apply_bitstring_arr)
        self.cache[id(gates.cswap)] = self.adapt(gates._cswap_register)

        def discard_array(qs):
            for q in qs:
                discard(q)

        def measure_array(qs):
            return StaticArray(measure(q) for q in qs)

        def collect_measurements(results):
            return [result.read() for result in results]

        import guppylang.std.quantum as quantum

        self.cache[id(quantum.discard_array)] = discard_array
        self.cache[id(quantum.measure_array)] = measure_array
        self.cache[id(quantum.collect_measurements)] = collect_measurements
