#ifndef CSR_GRAPH_H
#define CSR_GRAPH_H

#ifdef _WIN32
/* On Windows, avoid including Windows SDK stdio.h which requires vcruntime.h
   Use MinGW's stdio instead */
#ifndef __USE_MINGW_ANSI_STDIO
#define __USE_MINGW_ANSI_STDIO 1
#endif
#endif

#include <stdio.h>

// Compressed Sparse Row (CSR) graph structure
typedef struct {
    int num_nodes;      // Number of nodes in the graph
    int num_edges;      // Number of edges in the graph
    int *row_ptr;       // Row pointers (size = num_nodes + 1)
    int *col_ind;       // Column indices (size = num_edges)
} CSRGraph;

// Function to read a graph from a text file and create a CSR representation
void read_csr_graph(const char *filename, CSRGraph *graph);

// Function to free the memory allocated for the CSR graph
void free_csr_graph(CSRGraph *graph);

#endif // CSR_GRAPH_H