#include "csr_graph.h"
#include <stdlib.h>

void read_csr_graph(const char *filename, CSRGraph *graph) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        fprintf(stderr, "Error opening file: %s\n", filename);
        exit(EXIT_FAILURE);
    }

    // First pass: count nodes and edges
    int max_node_id = -1;
    int num_edges = 0;
    int from, to;
    char line[1000];
    while (fgets(line, sizeof(line), file)) {
        // Ignore lines starting with #
        if (line[0] == '#') continue;
        if (sscanf(line, "%d %d", &from, &to) == 2 || sscanf(line, "%d\t%d", &from, &to) == 2) {
            if (from > max_node_id) max_node_id = from;
            if (to > max_node_id) max_node_id = to;
            num_edges++;
        }
    }
    rewind(file);

    graph->num_nodes = max_node_id + 1;
    graph->num_edges = num_edges;
    graph->row_ptr = (int *)calloc(graph->num_nodes + 1, sizeof(int));
    graph->col_ind = (int *)malloc(graph->num_edges * sizeof(int));

    // Second pass: populate row_ptr
    while (fgets(line, sizeof(line), file)) {
        // Ignore lines starting with #
        if (line[0] == '#') continue;
        if (sscanf(line, "%d %d", &from, &to) == 2 || sscanf(line, "%d\t%d", &from, &to) == 2) {
            graph->row_ptr[from + 1]++;
        }
    }
    rewind(file);

    // Cumulative sum for row_ptr
    for (int i = 1; i <= graph->num_nodes; i++) {
        graph->row_ptr[i] += graph->row_ptr[i - 1];
    }

    // Third pass: populate col_ind
    int *temp_counts = (int *)calloc(graph->num_nodes, sizeof(int));
    while (fgets(line, sizeof(line), file)) {
        // Ignore lines starting with #
        if (line[0] == '#') continue;
        if (sscanf(line, "%d %d", &from, &to) == 2 || sscanf(line, "%d\t%d", &from, &to) == 2) {
            int index = graph->row_ptr[from] + temp_counts[from];
            graph->col_ind[index] = to;
            temp_counts[from]++;
        }
    }

    free(temp_counts);
    fclose(file);
}

void free_csr_graph(CSRGraph *graph) {
    if (graph) {
        free(graph->row_ptr);
        free(graph->col_ind);
    }
}