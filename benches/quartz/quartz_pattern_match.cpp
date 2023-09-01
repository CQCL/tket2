#include "quartz/context/context.h"
#include "quartz/parser/qasm_parser.h"
#include "quartz/tasograph/substitution.h"
#include "quartz/tasograph/tasograph.h"

using namespace quartz;

// template <class Tp> inline void black_box(Tp const &value) {
//   asm volatile("" : : "r,m"(value) : "memory");
// }

Context CTX({GateType::input_qubit, GateType::input_param, GateType::h,
             GateType::cx, GateType::x, GateType::t, GateType::tdg});

std::vector<std::string> get_sorted_qasm_files(const std::string &folder) {
  std::vector<std::filesystem::directory_entry> qasm_files;
  for (const auto &entry : std::filesystem::directory_iterator(folder)) {
    if (entry.is_regular_file() && entry.path().extension() == ".qasm") {
      qasm_files.push_back(entry);
    }
  }
  std::sort(qasm_files.begin(), qasm_files.end(),
            [](const std::filesystem::directory_entry &a,
               const std::filesystem::directory_entry &b) {
              return std::stoi(a.path().filename().stem()) <
                     std::stoi(b.path().filename().stem());
            });
  std::vector<std::string> qasm_file_names;
  for (const auto &entry : qasm_files) {
    qasm_file_names.push_back(entry.path());
  }
  return qasm_file_names;
}

// quartz::Context *load_context() {
//   return new Context({GateType::input_qubit, GateType::input_param,
//   GateType::h,
//                GateType::cx, GateType::x, GateType::t, GateType::tdg});
// }

// void free_context(quartz::Context *context) { delete context; }

Graph *load_graph(const char *file_name) {
  auto graph = Graph::from_qasm_file(&CTX, file_name);
  return new Graph(*graph);
}

void free_graph(const quartz::Graph *graph) { delete graph; }

quartz::GraphXfer **load_xfers(const char *const folder_name,
                               unsigned &n_xfers) {
  auto circs = std::vector<CircuitSeq *>();
  for (const auto &file_name : get_sorted_qasm_files(folder_name)) {
    auto circ = CircuitSeq::from_qasm_file(&CTX, file_name).release();
    assert(circ != nullptr);
    circs.push_back(circ);
  }
  n_xfers = circs.size();
  auto xfers = new GraphXfer *[n_xfers];
  for (unsigned i = 0; i < n_xfers; ++i) {
    auto circ = circs[i];
    auto empty_circ = new CircuitSeq(circ->get_num_qubits(),
                                     circ->get_num_input_parameters());
    auto xfer = GraphXfer::create_GraphXfer(&CTX, circ, empty_circ, true);
    assert(xfer != nullptr);
    xfers[i] = xfer;
    // free up memory
    delete empty_circ;
    delete circ;
    circ = nullptr;
  }
  std::cout << "number of xfers: " << n_xfers << std::endl;
  return xfers;
}

void free_xfers(quartz::GraphXfer **xfers, const unsigned n_xfers) {
  for (unsigned i = 0; i < n_xfers; ++i) {
    delete xfers[i];
  }
  delete[] xfers;
}

quartz::Op *get_ops(const quartz::Graph *const graph, unsigned &n_ops) {
  auto all_ops = std::vector<Op>();
  graph->topology_order_ops(all_ops);
  assert(all_ops.size() == (size_t)graph->gate_count());
  n_ops = all_ops.size();

  auto ops_c = new quartz::Op[all_ops.size()];
  std::memcpy(ops_c, all_ops.data(), all_ops.size() * sizeof(quartz::Op));
  return ops_c;
}
void free_ops(quartz::Op *ops) { delete[] ops; }

unsigned pattern_match(const quartz::Graph *const graph,
                       const quartz::Op *const ops, const unsigned n_ops,
                       quartz::GraphXfer *const *const xfers,
                       const unsigned n_xfers) {
  auto cnt = 0;
  for (unsigned i = 0; i < n_ops; ++i) {
    auto &op = ops[i];
    for (unsigned j = 0; j < n_xfers; ++j) {
      auto xfer = xfers[j];
      // pattern matching + convexity test
      cnt += graph->xfer_appliable(xfer, op);
    }
  }
  return cnt;
}

// std::vector<double> benchmark(std::shared_ptr<Graph> graph,
//                               const std::vector<GraphXfer *> &xfers) {
//   std::vector<double> timings;
//   for (int i = 200; i <= 4000; i += 200) {
//     std::cout << i << " patterns..." << std::endl;
//     timings.push_back(pattern_match(graph, xfers, i));
//   }
//   return timings;
// }

// void save_to_json(std::vector<double> &vec, const std::string &file_name) {
//   std::ofstream file(file_name);
//   file << "[";
//   for (size_t i = 0; i < vec.size(); ++i) {
//     file << vec[i];
//     if (i != vec.size() - 1) {
//       file << ",";
//     }
//   }
//   file << "]";
//   file.close();
// }

// int main() {
//   std::cout << "Benchmarking pattern matching" << std::endl;
//   auto graph =
//   load_graph("../experiment/circs/t_tdg_circs/barenco_tof_5.qasm"); auto
//   xfers =
//       load_xfers("../../tket2proto/test_files/T_Tdg_H_X_CX_complete_ECC_set");
//   auto timings = benchmark(graph, xfers);
//   save_to_json(timings, "pattern_matching.json");
//   std::cout << "Saved timings to pattern_matching.json" << std::endl;
//   return 0;
// }