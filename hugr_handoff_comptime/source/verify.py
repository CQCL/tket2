"""Check fixed classical outputs before and after compile-time expansion."""

import argparse
import hashlib
import json
from pathlib import Path

from guppyalgos.utils import int_to_bits
from guppylang.emulator import EmulatorBuilder
from hugr.package import Package

HERE = Path(__file__).resolve().parent.parent


def expected_outputs(record):
    inputs = record["fixed_inputs"]
    shapes = record["named_outputs"]
    if record["family"] == "qrom":
        n_address = int(shapes["address"][4:-1])
        n_data = int(shapes["data"][4:-1])
        return {
            "address": int_to_bits(inputs["address"], n_address),
            "data": int_to_bits((5 * inputs["address"] + 3) % 2**n_data, n_data),
        }
    if record["family"] == "select":
        return {
            "index": int_to_bits(1, int(shapes["index"][4:-1])),
            "state": [False, True],
        }
    width = int(shapes["a"][4:-1])
    expected = {"a": int_to_bits(inputs["a"], width)}
    if "sum" in shapes:
        total = inputs["a"] + inputs["b"]
        expected.update(
            sum=int_to_bits(total % 2**width, width), carry=total >= 2**width
        )
    else:
        expected["b"] = int_to_bits(inputs["b"], width)
        if "product" in shapes:
            expected["product"] = int_to_bits(
                inputs["a"] * inputs["b"] % 2**width, width
            )
        else:
            expected["target"] = True
    return expected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--structure-only", action="store_true")
    args = parser.parse_args()
    manifest = json.loads((HERE / "manifest.json").read_text())
    original = json.loads(
        (HERE.parent / "hugr_handoff_runtime/manifest.json").read_text()
    )
    assert manifest["versions"] == original["versions"]
    assert len(manifest["artifacts"]) == len(original["artifacts"]) == 19
    for record, reference in zip(
        manifest["artifacts"], original["artifacts"], strict=True
    ):
        for field in (
            "file",
            "family",
            "fixed_inputs",
            "named_outputs",
            "logical_qubits",
            "peak_qubit_budget",
        ):
            assert record[field] == reference[field], (record["file"], field)
        contents = (HERE / record["file"]).read_bytes()
        assert hashlib.sha256(contents).hexdigest() == record["sha256"]
        package = Package.from_str(contents.decode())
        for graph in package.modules:
            for node in graph:
                if type(graph[node].op).__name__ != "TailLoop":
                    continue
                for child in graph.descendants(node):
                    op = graph[child].op
                    assert type(op).__name__ not in {"Call", "CallIndirect"}
                    if type(op).__name__ == "ExtOp" and op.name().startswith(
                        "tket.quantum."
                    ):
                        assert op.name() == "tket.quantum.MeasureFree", (
                            record["file"],
                            op.name(),
                        )
    print(
        "Verified 19 inputs; remaining loops contain only terminal measurements",
        flush=True,
    )
    if args.structure_only:
        return
    records = []
    for record in manifest["artifacts"]:
        if record["family"] == "lcu" or record["peak_qubit_budget"] > 20:
            continue
        expected = expected_outputs(record)
        for directory in (HERE.parent / "hugr_handoff_runtime", HERE):
            package = Package.from_str((directory / record["file"]).read_text())
            emulator = EmulatorBuilder().build(package, record["peak_qubit_budget"])
            result = emulator.with_seed(42).with_shots(4).run()
            actual = [shot.as_dict() for shot in result.results]
            assert actual == [expected] * 4, (
                directory.name,
                record["file"],
                actual,
                expected,
            )
            print(f"Verified {directory.name}/{record['file']}", flush=True)
            records.append(
                {
                    "suite": directory.name,
                    "file": record["file"],
                    "sha256": hashlib.sha256(
                        (directory / record["file"]).read_bytes()
                    ).hexdigest(),
                    "shots": 4,
                }
            )
    (HERE / "execution_checks.json").write_text(json.dumps(records, indent=2) + "\n")


if __name__ == "__main__":
    main()
