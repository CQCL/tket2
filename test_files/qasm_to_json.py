"""
Converts all .qasm files in a folder to .json files using the
pytket.qasm.circuit_from_qasm converter.

Usage: python qasm_to_json.py <folder>
"""

import os
import sys
import json
from pytket.qasm import circuit_from_qasm

if len(sys.argv) != 2:
    print("Usage: python convert_qasm_to_json.py <folder>")
    sys.exit(1)

folder = sys.argv[1]

for filename in os.listdir(folder):
    if filename.endswith(".qasm"):
        input_path = os.path.join(folder, filename)
        output_path = os.path.join(folder, filename[:-5] + ".json")
        with open(output_path, "w", encoding="utf-8") as output_file:
            circuit = circuit_from_qasm(input_path)
            json.dump(circuit.to_dict(), output_file)