#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include "pagerank.h"
#include "graph.h"
#include "partition.h"
#include "communication.h"

int main(int argc, char** argv) {
    int rank, size;
    char* filename;
    double threshold;
    int max_iterations;

    // Initialize the MPI environment
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Handle command-line arguments
    if (argc < 4) {
        if (rank == 0) {
            fprintf(stderr, "Usage: %s <graph_file> <threshold> <max_iterations>\n", argv[0]);
        }
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    filename = argv[1];
    threshold = atof(argv[2]);
    max_iterations = atoi(argv[3]);

    // Load the graph and partition it
    Graph* graph = load_graph(filename);
    partition_graph(graph, rank, size);

    // Initialize PageRank values
    initialize_pagerank(graph);

    // Execute the PageRank algorithm
    execute_pagerank(graph, threshold, max_iterations, rank, size);

    // Clean up
    free_graph(graph);
    MPI_Finalize();
    return EXIT_SUCCESS;
}