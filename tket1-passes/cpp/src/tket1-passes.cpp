extern "C" {
#include "tket1-passes.h"
}

#include <cstring>

#include "tket/Circuit/Circuit.hpp"
#include "tket/Transformations/BasicOptimisation.hpp"
#include "tket/Transformations/CliffordResynthesis.hpp"

using namespace tket;
using json = nlohmann::json;

struct TketCircuit {
  Circuit circuit;
};

TketCircuit *tket_circuit_from_json(const char *json_str) {
  if (!json_str)
    return nullptr;
  try {
    TketCircuit *tc = new TketCircuit;
    tc->circuit = json::parse(json_str);
    return tc;
  } catch (...) {
    return nullptr;
  }
}

TketError tket_circuit_to_json(const TketCircuit *tc, char **json_str) {
  if (!tc || !json_str)
    return TKET_ERROR_NULL_POINTER;
  try {
    json j = tc->circuit;
    std::string s = j.dump();
    *json_str = (char *)malloc(s.size() + 1);
    std::strcpy(*json_str, s.c_str());
    return TKET_SUCCESS;
  } catch (...) {
    return TKET_ERROR_CIRCUIT_INVALID;
  }
}

void tket_circuit_destroy(TketCircuit *tc) { delete tc; }

void tket_free_string(char *str) { free(str); }

TketError tket_two_qubit_squash(TketCircuit *tc, TketTargetGate target_gate,
                                double cx_fidelity, bool allow_swaps) {
  if (!tc)
    return TKET_ERROR_NULL_POINTER;
  try {
    OpType target = (target_gate == TKET_TARGET_CX) ? OpType::CX : OpType::TK2;
    Transforms::two_qubit_squash(target, cx_fidelity, allow_swaps)
        .apply(tc->circuit);
    return TKET_SUCCESS;
  } catch (...) {
    return TKET_ERROR_UNKNOWN;
  }
}

TketError tket_clifford_resynthesis(TketCircuit *tc, bool allow_swaps) {
  if (!tc)
    return TKET_ERROR_NULL_POINTER;
  try {
    Transforms::clifford_resynthesis(std::nullopt, allow_swaps)
        .apply(tc->circuit);
    return TKET_SUCCESS;
  } catch (...) {
    return TKET_ERROR_UNKNOWN;
  }
}

const char *tket_error_string(TketError error) {
  switch (error) {
  case TKET_SUCCESS:
    return "Success";
  case TKET_ERROR_NULL_POINTER:
    return "Null pointer error";
  case TKET_ERROR_INVALID_ARGUMENT:
    return "Invalid argument error";
  case TKET_ERROR_CIRCUIT_INVALID:
    return "Circuit invalid error";
  case TKET_ERROR_MEMORY:
    return "Memory error";
  case TKET_ERROR_PARSE_JSON:
    return "JSON parse error";
  default:
    return "Unknown error";
  }
}
