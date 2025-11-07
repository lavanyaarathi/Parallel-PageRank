Sure, here's the contents for the file `/pagerank-mpi/pagerank-mpi/test/test_graph.c`:

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "../src/graph.h"

void test_load_graph() {
    Graph graph;
    load_graph("test_graph.txt", &graph);
    
    assert(graph.num_nodes > 0);
    assert(graph.adjacency_list != NULL);
    
    // Additional assertions can be added based on the expected graph structure
    printf("test_load_graph passed.\n");
}

void test_create_local_adjacency_list() {
    Graph graph;
    load_graph("test_graph.txt", &graph);
    
    int local_nodes = graph.num_nodes / 2; // Simulating a partition
    int *local_adjacency_list = create_local_adjacency_list(&graph, 0, local_nodes);
    
    assert(local_adjacency_list != NULL);
    // Additional assertions can be added to verify the local adjacency list
    printf("test_create_local_adjacency_list passed.\n");
}

int main() {
    test_load_graph();
    test_create_local_adjacency_list();
    
    printf("All graph tests passed.\n");
    return 0;
}