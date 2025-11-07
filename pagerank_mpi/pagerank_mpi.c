/********************************************************************/
/*    Pagerank project - MPI Distributed version                    */
/*    *based on Cleve Moler's matlab implementation               */
/*                                                                  */
/*    Implemented for distributed memory parallelism using MPI     */
/********************************************************************/

/******************** Includes - Defines ****************/
#include "pagerank_mpi.h"
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

/***** Read graph connections from txt file *****/
void Read_from_txt_file(char* filename, int rank, int size)
{
    FILE *fid;
    int from_idx, to_idx;
    char line[1000];
    
    // Calculate local partition range
    int nodes_per_proc = N / size;
    int remainder = N % size;
    int local_start = rank * nodes_per_proc + (rank < remainder ? rank : remainder);
    int local_end = local_start + nodes_per_proc + (rank < remainder ? 1 : 0);
    
    // Allocate memory for local nodes
    Nodes = (Node*) malloc((local_end - local_start) * sizeof(Node));
    for (int i = 0; i < (local_end - local_start); i++) {
        Nodes[i].con_size = 0;
        Nodes[i].To_id = (int*) malloc(sizeof(int));
    }
    
    // Only rank 0 reads the file and distributes data
    if (rank == 0) {
        fid = fopen(filename, "r");
        if (fid == NULL) {
            printf("Error opening data file\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        
        // First pass: count connections for each node
        int *global_con_sizes = (int*) calloc(N, sizeof(int));
        while (fgets(line, sizeof(line), fid)) {
            if (strncmp(line, "#", 1) != 0) {
                if (sscanf(line, "%d\t%d\n", &from_idx, &to_idx) == 2) {
                    global_con_sizes[from_idx]++;
                }
            }
        }
        rewind(fid);
        
        // Allocate global node structures
        Node *global_nodes = (Node*) malloc(N * sizeof(Node));
        for (int i = 0; i < N; i++) {
            global_nodes[i].con_size = 0;
            global_nodes[i].To_id = (int*) malloc(global_con_sizes[i] * sizeof(int));
        }
        
        // Second pass: read actual connections
        while (fgets(line, sizeof(line), fid)) {
            if (strncmp(line, "#", 1) != 0) {
                if (sscanf(line, "%d\t%d\n", &from_idx, &to_idx) == 2) {
                    int temp_size = global_nodes[from_idx].con_size;
                    global_nodes[from_idx].To_id[temp_size] = to_idx;
                    global_nodes[from_idx].con_size++;
                }
            }
        }
        fclose(fid);
        
        // Distribute data to other processes
        for (int dest = 1; dest < size; dest++) {
            int dest_start = dest * nodes_per_proc + (dest < remainder ? dest : remainder);
            int dest_end = dest_start + nodes_per_proc + (dest < remainder ? 1 : 0);
            int dest_size = dest_end - dest_start;
            
            for (int i = 0; i < dest_size; i++) {
                int global_idx = dest_start + i;
                MPI_Send(&global_nodes[global_idx].con_size, 1, MPI_INT, dest, 0, MPI_COMM_WORLD);
                if (global_nodes[global_idx].con_size > 0) {
                    MPI_Send(global_nodes[global_idx].To_id, global_nodes[global_idx].con_size, MPI_INT, dest, 1, MPI_COMM_WORLD);
                }
            }
        }
        
        // Copy local data for rank 0
        for (int i = 0; i < (local_end - local_start); i++) {
            int global_idx = local_start + i;
            Nodes[i].con_size = global_nodes[global_idx].con_size;
            free(Nodes[i].To_id);
            if (Nodes[i].con_size > 0) {
                Nodes[i].To_id = (int*) malloc(Nodes[i].con_size * sizeof(int));
                memcpy(Nodes[i].To_id, global_nodes[global_idx].To_id, Nodes[i].con_size * sizeof(int));
            } else {
                Nodes[i].To_id = NULL;
            }
        }
        
        // Cleanup global data
        for (int i = 0; i < N; i++) {
            free(global_nodes[i].To_id);
        }
        free(global_nodes);
        free(global_con_sizes);
        
    } else {
        // Receive data from rank 0
        for (int i = 0; i < (local_end - local_start); i++) {
            MPI_Recv(&Nodes[i].con_size, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            free(Nodes[i].To_id);
            if (Nodes[i].con_size > 0) {
                Nodes[i].To_id = (int*) malloc(Nodes[i].con_size * sizeof(int));
                MPI_Recv(Nodes[i].To_id, Nodes[i].con_size, MPI_INT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            } else {
                Nodes[i].To_id = NULL;
            }
        }
    }
    
    printf("Rank %d: Local nodes %d to %d, total local nodes: %d\n", 
           rank, local_start, local_end-1, local_end - local_start);
}

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

/***** Distributed PageRank algorithm *****/
void Distributed_PageRank(int rank, int size, int local_start, int local_end)
{
    int iterations = 0;
    double max_error = 1.0;
    int local_size = local_end - local_start;
    
    // Allocate global array for rank gathering
    if (rank == 0) {
        global_p_t1 = (double*) malloc(N * sizeof(double));
    }
    
    printf("Rank %d: Starting PageRank algorithm\n", rank);
    
    while (max_error > threshold)
    {
        double local_sum = 0.0;
        double local_max_error = -1.0;
        
        // Exchange rank values using non-blocking communication
        Exchange_Ranks_Nonblocking(rank, size, local_start, local_end);
        
        // Update local nodes
        for (int i = 0; i < local_size; i++)
        {
            Nodes[i].p_t0 = Nodes[i].p_t1;
            Nodes[i].p_t1 = 0.0;
        }
        
        // Compute contributions
        for (int i = 0; i < local_size; i++)
        {
            int global_node_id = local_start + i;
            
            if (Nodes[i].con_size != 0)
            {
                // Distribute rank to connected nodes
                for (int j = 0; j < Nodes[i].con_size; j++)
                {
                    int target_node = Nodes[i].To_id[j];
                    // Find which process owns this target node
                    int target_proc = target_node / (N / size);
                    if (target_proc >= size) target_proc = size - 1;
                    
                    // Add contribution to target node (using global array)
                    if (rank == target_proc) {
                        int local_target_idx = target_node - (target_proc * (N / size) + (target_proc < (N % size) ? target_proc : (N % size)));
                        Nodes[local_target_idx].p_t1 += Nodes[i].p_t0 / Nodes[i].con_size;
                    }
                }
            }
            else
            {
                // Node with no outgoing connections contributes to all
                local_sum += Nodes[i].p_t0 / N;
            }
        }
        
        // Compute global sum
        double global_sum = Compute_Global_Sum(local_sum, rank, size);
        
        // Update probabilities and compute local max error
        for (int i = 0; i < local_size; i++)
        {
            Nodes[i].p_t1 = d * (Nodes[i].p_t1 + global_sum) + (1 - d) * Nodes[i].e;
            
            double error = fabs(Nodes[i].p_t1 - Nodes[i].p_t0);
            if (error > local_max_error) {
                local_max_error = error;
            }
        }
        
        // Compute global max error
        max_error = Compute_Global_Max_Error(local_max_error, rank, size);
        
        if (rank == 0) {
            printf("Iteration %d: Max Error = %f\n", iterations + 1, max_error);
        }
        
        iterations++;
        
        // Safety check to prevent infinite loops
        if (iterations > 1000) {
            if (rank == 0) printf("Warning: Maximum iterations reached\n");
            break;
        }
    }
    
    if (rank == 0) {
        printf("Total iterations: %d\n", iterations);
        free(global_p_t1);
    }
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
    
    // Calculate local partition
    int nodes_per_proc = N / size;
    int remainder = N % size;
    int local_start = rank * nodes_per_proc + (rank < remainder ? rank : remainder);
    int local_end = local_start + nodes_per_proc + (rank < remainder ? 1 : 0);
    
    // Read graph data
    Read_from_txt_file(filename, rank, size);
    
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
    Distributed_PageRank(rank, size, local_start, local_end);
    
    gettimeofday(&end, NULL);
    
    totaltime = (((end.tv_usec - start.tv_usec) / 1.0e6 + end.tv_sec - start.tv_sec) * 1000) / 1000;
    
    if (rank == 0) {
        printf("\nTotal time = %f seconds\n", totaltime);
        printf("End of program!\n");
    }
    
    // Cleanup
    int local_size = local_end - local_start;
    for (int i = 0; i < local_size; i++) {
        if (Nodes[i].To_id != NULL) {
            free(Nodes[i].To_id);
        }
    }
    free(Nodes);
    
    // Finalize MPI
    MPI_Finalize();
    
    return EXIT_SUCCESS;
}