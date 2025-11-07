#ifndef PARTITION_H
#define PARTITION_H

#include "graph.h"

// Function to partition the graph into subgraphs for each process
void partition_graph(Graph *graph, int num_processes, Graph *local_graph);

#endif // PARTITION_H