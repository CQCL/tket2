namespace quartz {

class Graph;
class GraphXfer;
class Op;

} // namespace quartz

/** Perform pattern matching and return match count.
 *
 * @param graph The graph to match on
 * @param ops The ops of graph in topological order
 * @param n_ops The number of ops in the graph
 * @param xfers The patterns to match
 * @param n_xfers The number of patterns to match
 */
unsigned pattern_match(const quartz::Graph *const graph,
                       const quartz::Op *const ops, const unsigned n_ops,
                       quartz::GraphXfer *const *const xfers,
                       const unsigned n_xfers);

quartz::Graph *load_graph(const char *file_name);
void free_graph(const quartz::Graph *graph);

/** Get all ops in graph in topological order. */
quartz::Op *get_ops(const quartz::Graph *const graph, unsigned &n_ops);
void free_ops(quartz::Op *ops);

/** Load all patterns in folder. */
quartz::GraphXfer **load_xfers(const char *const folder_name,
                               unsigned &n_xfers);
void free_xfers(quartz::GraphXfer **xfers, const unsigned n_xfers);