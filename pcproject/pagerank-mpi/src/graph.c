Sure, here's the contents for the file `/pagerank-mpi/pagerank-mpi/src/graph.c`:

#include "graph.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>

Graph* create_graph(int num_nodes) {
    Graph* graph = (Graph*)malloc(sizeof(Graph));
    graph->num_nodes = num_nodes;
    graph->adjacency_list = (Node**)malloc(num_nodes * sizeof(Node*));
    
    for (int i = 0; i < num_nodes; i++) {
        graph->adjacency_list[i] = NULL;
    }
    
    return graph;
}

void free_graph(Graph* graph) {
    for (int i = 0; i < graph->num_nodes; i++) {
        Node* current = graph->adjacency_list[i];
        while (current != NULL) {
            Node* temp = current;
            current = current->next;
            free(temp);
        }
    }
    free(graph->adjacency_list);
    free(graph);
}

void load_graph_from_file(Graph* graph, const char* filename) {
    FILE* file = fopen(filename, "r");
    if (file == NULL) {
        fprintf(stderr, "Error opening file: %s\n", filename);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    int from, to;
    while (fscanf(file, "%d %d", &from, &to) == 2) {
        Node* new_node = (Node*)malloc(sizeof(Node));
        new_node->id = to;
        new_node->next = graph->adjacency_list[from];
        graph->adjacency_list[from] = new_node;
    }

    fclose(file);
}

void create_local_adjacency_list(Graph* global_graph, Graph* local_graph, int start, int end) {
    for (int i = start; i < end; i++) {
        local_graph->adjacency_list[i] = global_graph->adjacency_list[i];
    }
}