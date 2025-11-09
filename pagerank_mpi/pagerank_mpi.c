/********************************************************************/
/*    Pagerank project - MPI Distributed version (CSR)              */
/*    *based on Cleve Moler's matlab implementation               */
/*                                                                  */
/*    Implemented for distributed memory parallelism using MPI     */
/********************************************************************/

/******************** Includes - Defines ****************/
#include "pagerank_mpi.h"
#include "csr_graph.h"
#ifdef _WIN32
#ifndef __USE_MINGW_ANSI_STDIO
#define __USE_MINGW_ANSI_STDIO 1
#endif
#endif
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <math.h>
#include <assert.h>
#include <string.h>
#include <mpi.h>

/******************** Global Variables ****************/
// Number of nodes
int N;

// Convergence threshold and algorithm's parameter d  
double threshold, d;

// Table of node's data (local partition)
Node *Nodes;

// Global arrays for gathering results
double *global_p_t1;

// MPI data types
MPI_Datatype mpi_node_type;

/******************** Function Implementations ****************/

/***** Create P and E with equal probability *****/
void Random_P_E(int local_start, int local_end)
{
    int local_size = local_end - local_start;
    
    for (int i = 0; i < local_size; i++)
    {
        Nodes[i].p_t0 = 0;
        Nodes[i].p_t1 = 1.0 / N;
        Nodes[i].e = 1.0 / N;
    }
}

/***** Exchange ranks using non-blocking communication *****/
void Exchange_Ranks_Nonblocking(int rank, int size, int local_start, int local_end)
{
    int local_size = local_end - local_start;
    
    // Allocate buffers for non-blocking communication
    double *outgoing_ranks = (double*) malloc(local_size * sizeof(double));
    double *incoming_buffer = (double*) malloc(N * sizeof(double));
    
    // Prepare outgoing data
    for (int i = 0; i < local_size; i++) {
        outgoing_ranks[i] = Nodes[i].p_t0;
    }
    
    // Array to store MPI requests for non-blocking operations
    MPI_Request *requests = (MPI_Request*) malloc(2 * size * sizeof(MPI_Request));
    int request_count = 0;
    
    // Non-blocking send to all processes
    for (int dest = 0; dest < size; dest++) {
        if (dest != rank) {
            MPI_Isend(outgoing_ranks, local_size, MPI_DOUBLE, dest, 0, MPI_COMM_WORLD, 
                     &requests[request_count++]);
        }
    }
    
    // Non-blocking receive from all processes
    int recv_count = 0;
    for (int src = 0; src < size; src++) {
        if (src != rank) {
            // Calculate source process range
            int src_nodes_per_proc = N / size;
            int src_remainder = N % size;
            int src_start = src * src_nodes_per_proc + (src < src_remainder ? src : src_remainder);
            int src_size = src_nodes_per_proc + (src < src_remainder ? 1 : 0);
            
            MPI_Irecv(incoming_buffer + src_start, src_size, MPI_DOUBLE, src, 0, 
                     MPI_COMM_WORLD, &requests[request_count++]);
            recv_count++;
        }
    }
    
    // Copy local data to incoming buffer
    memcpy(incoming_buffer + local_start, outgoing_ranks, local_size * sizeof(double));
    
    // Wait for all non-blocking operations to complete
    MPI_Waitall(request_count, requests, MPI_STATUSES_IGNORE);
    
    // Copy received data to global array
    if (global_p_t1 != NULL) {
        memcpy(global_p_t1, incoming_buffer, N * sizeof(double));
    }
    
    // Cleanup
    free(outgoing_ranks);
    free(incoming_buffer);
    free(requests);
}

/***** Compute global max error *****/
double Compute_Global_Max_Error(double local_max_error, int rank, int size)
{
    double global_max_error;
    MPI_Allreduce(&local_max_error, &global_max_error, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    return global_max_error;
}

/***** Compute global sum *****/
double Compute_Global_Sum(double local_sum, int rank, int size)
{
    double global_sum;
    MPI_Allreduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    return global_sum;
}

/***** Distributed PageRank algorithm using CSR *****/
void Distributed_PageRank_csr(int rank, int size, int local_start, int local_end, const CSRGraph *graph)
{
    int iterations = 0;
    double max_error = 1.0;
    int local_size = local_end - local_start;

    double *p_t0_global = (double *)malloc(N * sizeof(double));
    double *p_t1_local = (double *)calloc(local_size, sizeof(double));

    while (max_error > threshold) {
        // Gather all local p_t1 values to form global p_t0
        double *local_p_t1 = (double *)malloc(local_size * sizeof(double));
        for (int i = 0; i < local_size; i++) {
            local_p_t1[i] = Nodes[i].p_t1;
        }
        
        // Calculate displacements and counts for each process
        int *recvcounts = (int *)malloc(size * sizeof(int));
        int *displs = (int *)malloc(size * sizeof(int));
        for (int i = 0; i < size; i++) {
            int proc_nodes = N / size;
            int proc_remainder = N % size;
            recvcounts[i] = proc_nodes + (i < proc_remainder ? 1 : 0);
            displs[i] = (i == 0) ? 0 : displs[i-1] + recvcounts[i-1];
        }
        
        // Gather all local PageRank values to all processes
        MPI_Allgatherv(local_p_t1, local_size, MPI_DOUBLE, 
                      p_t0_global, recvcounts, displs, MPI_DOUBLE, MPI_COMM_WORLD);
        
        free(local_p_t1);
        free(recvcounts);
        free(displs);

        // Compute dangling node sum in parallel
        double local_dangling_sum = 0.0;
        for (int i = local_start; i < local_end; i++) {
            if (graph->row_ptr[i+1] == graph->row_ptr[i]) {
                local_dangling_sum += p_t0_global[i];
            }
        }
        double dangling_sum;
        MPI_Allreduce(&local_dangling_sum, &dangling_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        // Optimized CSR matrix-vector multiplication
        // For each node j, distribute its PageRank to its neighbors
        for (int j = 0; j < N; j++) {
            int out_degree = graph->row_ptr[j+1] - graph->row_ptr[j];
            if (out_degree > 0) {
                double contribution = p_t0_global[j] / out_degree;
                // For each neighbor of j
                for (int k = graph->row_ptr[j]; k < graph->row_ptr[j+1]; k++) {
                    int neighbor = graph->col_ind[k];
                    // Check if this neighbor is in our local partition
                    if (neighbor >= local_start && neighbor < local_end) {
                        int local_idx = neighbor - local_start;
                        p_t1_local[local_idx] += contribution;
                    }
                }
            }
        }

        for (int i = 0; i < local_size; i++) {
            Nodes[i].p_t1 = d * (p_t1_local[i] + dangling_sum / N) + (1.0 - d) / N;
        }
        
        // Reset p_t1_local for next iteration
        memset(p_t1_local, 0, local_size * sizeof(double));

        double local_max_error = 0.0;
        double local_l1_norm = 0.0;
        for (int i = 0; i < local_size; i++) {
            double error = fabs(Nodes[i].p_t1 - p_t0_global[local_start + i]);
            local_l1_norm += error;
            if (error > local_max_error) {
                local_max_error = error;
            }
        }

        double global_l1_norm;
        MPI_Allreduce(&local_max_error, &max_error, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        MPI_Allreduce(&local_l1_norm, &global_l1_norm, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        if (rank == 0) {
            printf("Iteration %d, Max Error: %f, L1 Norm: %f\n", iterations, max_error, global_l1_norm);
        }
        iterations++;
        
        // Use L1 norm for convergence check (more robust)
        if (global_l1_norm < threshold) {
            if (rank == 0) {
                printf("Converged based on L1 norm: %f < %f\n", global_l1_norm, threshold);
            }
            break;
        }
    }

    free(p_t0_global);
    free(p_t1_local);
}

/***** Block-based PageRank with periodic synchronization *****/
void Distributed_PageRank_Block(int rank, int size, int local_start, int local_end, const CSRGraph *graph, int sync_interval)
{
    int iterations = 0;
    double max_error = 1.0;
    int local_size = local_end - local_start;

    double *p_t0_global = (double *)malloc(N * sizeof(double));
    double *p_t1_local = (double *)calloc(local_size, sizeof(double));
    double *p_t0_local = (double *)malloc(local_size * sizeof(double));

    // Initialize local p_t0 from global
    for (int i = 0; i < local_size; i++) {
        p_t0_local[i] = Nodes[i].p_t1;
    }

    while (max_error > threshold) {
        // Perform local iterations without synchronization
        for (int local_iter = 0; local_iter < sync_interval; local_iter++) {
            // Local block computation
            for (int i = 0; i < local_size; i++) {
                p_t1_local[i] = 0.0;
            }
            
            // Compute contributions from nodes in our local block
            for (int j = local_start; j < local_end; j++) {
                int out_degree = graph->row_ptr[j+1] - graph->row_ptr[j];
                if (out_degree > 0) {
                    int local_j = j - local_start;
                    double contribution = p_t0_local[local_j] / out_degree;
                    for (int k = graph->row_ptr[j]; k < graph->row_ptr[j+1]; k++) {
                        int neighbor = graph->col_ind[k];
                        if (neighbor >= local_start && neighbor < local_end) {
                            int local_idx = neighbor - local_start;
                            p_t1_local[local_idx] += contribution;
                        }
                    }
                }
            }
            
            // Update local PageRank values
            for (int i = 0; i < local_size; i++) {
                p_t0_local[i] = d * p_t1_local[i] + (1.0 - d) / N;
                p_t1_local[i] = 0.0; // Reset for next iteration
            }
        }
        
        // Synchronize: gather all local values
        double *local_p_t1 = (double *)malloc(local_size * sizeof(double));
        for (int i = 0; i < local_size; i++) {
            local_p_t1[i] = p_t0_local[i];
        }
        
        int *recvcounts = (int *)malloc(size * sizeof(int));
        int *displs = (int *)malloc(size * sizeof(int));
        for (int i = 0; i < size; i++) {
            int proc_nodes = N / size;
            int proc_remainder = N % size;
            recvcounts[i] = proc_nodes + (i < proc_remainder ? 1 : 0);
            displs[i] = (i == 0) ? 0 : displs[i-1] + recvcounts[i-1];
        }
        
        MPI_Allgatherv(local_p_t1, local_size, MPI_DOUBLE, 
                      p_t0_global, recvcounts, displs, MPI_DOUBLE, MPI_COMM_WORLD);
        
        free(local_p_t1);
        free(recvcounts);
        free(displs);
        
        // Compute dangling sum in parallel
        double local_dangling_sum = 0.0;
        for (int i = local_start; i < local_end; i++) {
            if (graph->row_ptr[i+1] == graph->row_ptr[i]) {
                local_dangling_sum += p_t0_global[i];
            }
        }
        double dangling_sum;
        MPI_Allreduce(&local_dangling_sum, &dangling_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        
        // Update with contributions from all blocks
        for (int j = 0; j < N; j++) {
            int out_degree = graph->row_ptr[j+1] - graph->row_ptr[j];
            if (out_degree > 0) {
                double contribution = p_t0_global[j] / out_degree;
                for (int k = graph->row_ptr[j]; k < graph->row_ptr[j+1]; k++) {
                    int neighbor = graph->col_ind[k];
                    if (neighbor >= local_start && neighbor < local_end) {
                        int local_idx = neighbor - local_start;
                        p_t1_local[local_idx] += contribution;
                    }
                }
            }
        }
        
        // Final update with dangling nodes
        for (int i = 0; i < local_size; i++) {
            Nodes[i].p_t1 = d * (p_t1_local[i] + dangling_sum / N) + (1.0 - d) / N;
            p_t0_local[i] = Nodes[i].p_t1;
            p_t1_local[i] = 0.0;
        }
        
        // Compute convergence metrics
        double local_max_error = 0.0;
        double local_l1_norm = 0.0;
        for (int i = 0; i < local_size; i++) {
            double error = fabs(Nodes[i].p_t1 - p_t0_global[local_start + i]);
            local_l1_norm += error;
            if (error > local_max_error) {
                local_max_error = error;
            }
        }
        
        double global_l1_norm;
        MPI_Allreduce(&local_max_error, &max_error, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        MPI_Allreduce(&local_l1_norm, &global_l1_norm, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        
        if (rank == 0) {
            printf("Iteration %d, Max Error: %f, L1 Norm: %f\n", iterations, max_error, global_l1_norm);
        }
        iterations++;
        
        if (global_l1_norm < threshold) {
            if (rank == 0) {
                printf("Converged based on L1 norm: %f < %f\n", global_l1_norm, threshold);
            }
            break;
        }
    }
    
    free(p_t0_global);
    free(p_t1_local);
    free(p_t0_local);
}


/***** Main function *****/
int main(int argc, char** argv)
{
    int rank, size;
    double totaltime;
    struct timeval start, end;
    
    // Initialize MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    // Check input arguments
    if (argc < 5)
    {
        if (rank == 0) {
            printf("Error in arguments! Four arguments required: graph filename, N, threshold and d\n");
        }
        MPI_Finalize();
        return 0;
    }
    
    // Get arguments
    char filename[256];
    strcpy(filename, argv[1]);
    N = atoi(argv[2]);
    threshold = atof(argv[3]);
    d = atof(argv[4]);

    CSRGraph graph;
    if (rank == 0) {
        read_csr_graph(filename, &graph);
        N = graph.num_nodes;
    }

    // Broadcast graph dimensions
    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&graph.num_edges, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Allocate memory on other processes
    if (rank != 0) {
        graph.row_ptr = (int *)malloc((N + 1) * sizeof(int));
        graph.col_ind = (int *)malloc(graph.num_edges * sizeof(int));
    }

    // Broadcast CSR arrays
    MPI_Bcast(graph.row_ptr, N + 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(graph.col_ind, graph.num_edges, MPI_INT, 0, MPI_COMM_WORLD);
    
    // Calculate local partition
    int nodes_per_proc = N / size;
    int remainder = N % size;
    int local_start = rank * nodes_per_proc + (rank < remainder ? rank : remainder);
    int local_end = local_start + nodes_per_proc + (rank < remainder ? 1 : 0);

    Nodes = (Node*) malloc((local_end - local_start) * sizeof(Node));
    
    // Initialize probabilities
    Random_P_E(local_start, local_end);
    
    // Synchronize before starting timer
    MPI_Barrier(MPI_COMM_WORLD);
    
    if (rank == 0) {
        printf("\nMPI Distributed version of PageRank\n");
        printf("Using %d processes\n", size);
    }
    
    gettimeofday(&start, NULL);
    
    // Run distributed PageRank
    Distributed_PageRank_csr(rank, size, local_start, local_end, &graph);
    
    gettimeofday(&end, NULL);
    
    totaltime = (((end.tv_usec - start.tv_usec) / 1.0e6 + end.tv_sec - start.tv_sec) * 1000) / 1000;
    
    if (rank == 0) {
        printf("\nTotal time = %f seconds\n", totaltime);
        printf("End of program!\n");
    }
    
    // Cleanup
    free(Nodes);
    free_csr_graph(&graph);
    
    // Finalize MPI
    MPI_Finalize();
    
    return EXIT_SUCCESS;
}