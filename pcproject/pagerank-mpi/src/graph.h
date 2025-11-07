#ifndef GRAPH_H
#define GRAPH_H

#include <stdio.h>
#include <stdlib.h>
#include "types.h"

typedef struct {
    int num_nodes;
    int *adjacency_list; // Array of adjacency lists
    int *out_degree;     // Out-degree for each node
} Graph;

// Function prototypes
Graph* create_graph(int num_nodes);
void load_graph_from_file(Graph *graph, const char *filename);
void free_graph(Graph *graph);
void create_local_adjacency_list(Graph *graph, int rank, int size);

#endif // GRAPH_H