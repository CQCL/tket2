#include "quartz/context/context.h"
#include "quartz/parser/qasm_parser.h"
#include "quartz/tasograph/substitution.h"
#include "quartz/tasograph/tasograph.h"

using namespace quartz;

template <class Tp>
inline void black_box(Tp const& value) {
  asm volatile("" : : "r,m"(value) : "memory");
}

void ecc_to_qasm() {
  Context ctx({GateType::input_qubit, GateType::input_param, GateType::h,
               GateType::cx, GateType::x, GateType::t, GateType::tdg});

  EquivalenceSet eqs;
  // Load equivalent dags from file
  if (!eqs.load_json(&ctx, "../T_Tdg_H_X_CX_complete_ECC_set.json")) {
    std::cout << "Failed to load equivalence file." << std::endl;
    assert(false);
  }

  // Get xfer from the equivalent set
  auto ecc = eqs.get_all_equivalence_sets();
  std::vector<GraphXfer *> xfers;
  unsigned i = 0;
  for (auto eqcs : ecc) {
    for (auto circ : eqcs) {
        circ->to_qasm_file(&ctx, "../T_Tdg_H_X_CX_complete_ECC_set/" + std::to_string(i) + ".qasm");
        i += 1;
    }
  }
}

int main() {
  std::cout << "Benchmarking pattern matching" << std::endl;
  ecc_to_qasm();
}