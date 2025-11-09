#ifndef PAGERANK_MPI_H
#define PAGERANK_MPI_H

#include <mpi.h>
#include "csr_graph.h"

/******************** Structs - Defines ********************/

// Struct for a single node
typedef struct {
    double p_t0; // PageRank at time t
    double p_t1; // PageRank at time t+1
    double e;    // Dangling node contribution
} Node;

/******************** Function Prototypes ********************/

// PageRank algorithm
void Distributed_PageRank_csr(int rank, int size, int local_start, int local_end, const CSRGraph *graph);

// Block-based PageRank with periodic synchronization
void Distributed_PageRank_Block(int rank, int size, int local_start, int local_end, const CSRGraph *graph, int sync_interval);

// Initialize probabilities
void Random_P_E(int local_start, int local_end);

// Exchange ranks between processes
void Exchange_Ranks_Nonblocking(int rank, int size, int local_start, int local_end);

// Compute global max error
double Compute_Global_Max_Error(double local_max_error, int rank, int size);

// Compute global sum
double Compute_Global_Sum(double local_sum, int rank, int size);

#endif // PAGERANK_MPI_H