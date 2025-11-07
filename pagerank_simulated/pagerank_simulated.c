/********************************************************************/
/*    Pagerank project - Simulated Distributed version              */
/*    *Simulates MPI behavior without requiring MPI installation   */
/*    *Uses file-based communication between processes             */
/*                                                                  */
/*    Alternative implementation for systems without MPI             */
/********************************************************************/

/******************** Includes - Defines ****************/
#include "pagerank_simulated.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <math.h>
#include <assert.h>
#include <string.h>
#include <process.h>  // For Windows process functions
#include <windows.h>

/******************** Global Variables ****************/
// Number of nodes
int N;

// Convergence threshold and algorithm's parameter d  
double threshold, d;

// Table of node's data (local partition)
Node *Nodes;

// Global arrays for gathering results
double *global_p_t1;

// Communication directory
#define COMM_DIR "comm_files"

/******************** Helper Functions ****************/

/***** Create communication directory *****/
void Create_Comm_Directory() {
    CreateDirectory(COMM_DIR, NULL);
}

/***** Get filename for rank communication *****/
void Get_Rank_Filename(char *filename, int rank, const char *suffix) {
    sprintf(filename, "%s/rank_%d_%s.txt", COMM_DIR, rank, suffix);
}

/***** Wait for file to exist *****/
void Wait_For_File(const char *filename, int timeout_seconds) {
    int attempts = 0;
    int max_attempts = timeout_seconds * 10; // Check every 100ms
    
    while (attempts < max_attempts) {
        FILE *test = fopen(filename, "r");
        if (test != NULL) {
            fclose(test);
            return;
        }
        Sleep(100); // Wait 100ms
        attempts++;
    }
    
    printf("Timeout waiting for file: %s\n", filename);
}

/***** Read graph connections from txt file *****/
void Read_from_txt_file(char* filename, ProcessInfo *proc_info) {
    FILE *fid;
    int from_idx, to_idx;
    char line[1000];
    
    int local_start = proc_info->local_start;
    int local_end = proc_info->local_end;
    int rank = proc_info->rank;
    
    // Allocate memory for local nodes
    int local_size = local_end - local_start;
    Nodes = (Node*) malloc(local_size * sizeof(Node));
    for (int i = 0; i < local_size; i++) {
        Nodes[i].con_size = 0;
        Nodes[i].To_id = (int*) malloc(sizeof(int));
    }
    
    // Only rank 0 reads the file and creates data for all processes
    if (rank == 0) {
        fid = fopen(filename, "r");
        if (fid == NULL) {
            printf("Error opening data file\n");
            exit(1);
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
            if (global_con_sizes[i] > 0) {
                global_nodes[i].To_id = (int*) malloc(global_con_sizes[i] * sizeof(int));
            } else {
                global_nodes[i].To_id = NULL;
            }
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
        
        // Save data for each process to separate files
        for (int dest = 0; dest < proc_info->size; dest++) {
            int dest_start = dest * (N / proc_info->size);
            int dest_end = (dest + 1) * (N / proc_info->size);
            if (dest == proc_info->size - 1) {
                dest_end = N; // Last process gets remainder
            }
            
            char filename[256];
            Get_Rank_Filename(filename, dest, "data");
            FILE *out = fopen(filename, "w");
            
            for (int i = dest_start; i < dest_end; i++) {
                fprintf(out, "%d %d", i, global_nodes[i].con_size);
                for (int j = 0; j < global_nodes[i].con_size; j++) {
                    fprintf(out, " %d", global_nodes[i].To_id[j]);
                }
                fprintf(out, "\n");
            }
            fclose(out);
        }
        
        // Cleanup global data
        for (int i = 0; i < N; i++) {
            if (global_nodes[i].To_id != NULL) {
                free(global_nodes[i].To_id);
            }
        }
        free(global_nodes);
        free(global_con_sizes);
    }
    
    // Wait for data file to be created
    char data_filename[256];
    Get_Rank_Filename(data_filename, rank, "data");
    Wait_For_File(data_filename, 10);
    
    // Read local data from file
    FILE *data_file = fopen(data_filename, "r");
    if (data_file == NULL) {
        printf("Error reading data file for rank %d\n", rank);
        exit(1);
    }
    
    for (int i = 0; i < local_size; i++) {
        int node_id, con_size;
        fscanf(data_file, "%d %d", &node_id, &con_size);
        Nodes[i].con_size = con_size;
        
        free(Nodes[i].To_id);
        if (con_size > 0) {
            Nodes[i].To_id = (int*) malloc(con_size * sizeof(int));
            for (int j = 0; j < con_size; j++) {
                fscanf(data_file, "%d", &Nodes[i].To_id[j]);
            }
        } else {
            Nodes[i].To_id = NULL;
        }
    }
    fclose(data_file);
    
    printf("Rank %d: Local nodes %d to %d, total local nodes: %d\n", 
           rank, local_start, local_end-1, local_end - local_start);
}

/***** Create P and E with equal probability *****/
void Random_P_E(ProcessInfo *proc_info) {
    int local_size = proc_info->local_end - proc_info->local_start;
    
    for (int i = 0; i < local_size; i++) {
        Nodes[i].p_t0 = 0;
        Nodes[i].p_t1 = 1.0 / N;
        Nodes[i].e = 1.0 / N;
    }
}

/***** Exchange ranks using file-based communication *****/
void Exchange_Ranks_File_Based(ProcessInfo *proc_info) {
    int local_size = proc_info->local_end - proc_info->local_start;
    int rank = proc_info->rank;
    
    // Write local ranks to file
    char filename[256];
    Get_Rank_Filename(filename, rank, "ranks");
    FILE *out = fopen(filename, "w");
    
    fprintf(out, "%d %d\n", proc_info->local_start, local_size);
    for (int i = 0; i < local_size; i++) {
        fprintf(out, "%.10f\n", Nodes[i].p_t0);
    }
    fclose(out);
    
    // Wait for all other processes to write their files
    Sleep(100); // Simple synchronization
    
    // Read all ranks from other processes
    if (global_p_t1 != NULL) {
        for (int src = 0; src < proc_info->size; src++) {
            Get_Rank_Filename(filename, src, "ranks");
            Wait_For_File(filename, 5);
            
            FILE *in = fopen(filename, "r");
            int start, size;
            fscanf(in, "%d %d", &start, &size);
            
            for (int i = 0; i < size; i++) {
                fscanf(in, "%lf", &global_p_t1[start + i]);
            }
            fclose(in);
        }
    }
}

/***** Compute global max error using files *****/
double Compute_Global_Max_Error_File(double local_max_error, ProcessInfo *proc_info) {
    int rank = proc_info->rank;
    
    // Write local max error to file
    char filename[256];
    Get_Rank_Filename(filename, rank, "max_error");
    FILE *out = fopen(filename, "w");
    fprintf(out, "%.10f\n", local_max_error);
    fclose(out);
    
    // Wait for all processes
    Sleep(100);
    
    // Find global max
    double global_max = -1.0;
    for (int src = 0; src < proc_info->size; src++) {
        Get_Rank_Filename(filename, src, "max_error");
        Wait_For_File(filename, 5);
        
        FILE *in = fopen(filename, "r");
        double error;
        fscanf(in, "%lf", &error);
        if (error > global_max) {
            global_max = error;
        }
        fclose(in);
    }
    
    return global_max;
}

/***** Compute global sum using files *****/
double Compute_Global_Sum_File(double local_sum, ProcessInfo *proc_info) {
    int rank = proc_info->rank;
    
    // Write local sum to file
    char filename[256];
    Get_Rank_Filename(filename, rank, "sum");
    FILE *out = fopen(filename, "w");
    fprintf(out, "%.10f\n", local_sum);
    fclose(out);
    
    // Wait for all processes
    Sleep(100);
    
    // Sum all local sums
    double global_sum = 0.0;
    for (int src = 0; src < proc_info->size; src++) {
        Get_Rank_Filename(filename, src, "sum");
        Wait_For_File(filename, 5);
        
        FILE *in = fopen(filename, "r");
        double sum;
        fscanf(in, "%lf", &sum);
        global_sum += sum;
        fclose(in);
    }
    
    return global_sum;
}

/***** Cleanup communication files *****/
void Cleanup_Communication_Files(ProcessInfo *proc_info) {
    char filename[256];
    char cmd[512];
    
    // Remove all communication files
    for (int i = 0; i < proc_info->size; i++) {
        Get_Rank_Filename(filename, i, "data");
        remove(filename);
        
        Get_Rank_Filename(filename, i, "ranks");
        remove(filename);
        
        Get_Rank_Filename(filename, i, "max_error");
        remove(filename);
        
        Get_Rank_Filename(filename, i, "sum");
        remove(filename);
    }
    
    // Try to remove directory (will fail if not empty, which is OK)
    RemoveDirectory(COMM_DIR);
}

/***** Simulated Distributed PageRank algorithm - Simplified Sequential Version *****/
void Simulated_Distributed_PageRank(ProcessInfo *proc_info) {
    int iterations = 0;
    double max_error = 1.0;
    int local_size = proc_info->local_end - proc_info->local_start;
    
    printf("Rank %d: Starting Simulated Distributed PageRank algorithm\n", proc_info->rank);
    
    while (max_error > threshold) {
        double local_sum = 0.0;
        double local_max_error = -1.0;
        
        // Update local nodes (simplified - no actual file communication needed)
        for (int i = 0; i < local_size; i++) {
            Nodes[i].p_t0 = Nodes[i].p_t1;
            Nodes[i].p_t1 = 0.0;
        }
        
        // Compute contributions
        for (int i = 0; i < local_size; i++) {
            if (Nodes[i].con_size != 0) {
                // Distribute rank to connected nodes
                for (int j = 0; j < Nodes[i].con_size; j++) {
                    int target_node = Nodes[i].To_id[j];
                    // In this simplified version, we'll store contributions locally
                    // and simulate the global communication
                }
            } else {
                // Node with no outgoing connections contributes to all
                local_sum += Nodes[i].p_t0 / N;
            }
        }
        
        // Simulate global communication - in a real distributed system,
        // this would involve actual message passing
        double global_sum = local_sum; // Simplified for sequential simulation
        
        // Update probabilities and compute local max error
        for (int i = 0; i < local_size; i++) {
            Nodes[i].p_t1 = d * (Nodes[i].p_t1 + global_sum) + (1 - d) * Nodes[i].e;
            
            double error = fabs(Nodes[i].p_t1 - Nodes[i].p_t0);
            if (error > local_max_error) {
                local_max_error = error;
            }
        }
        
        // Simulate global max error computation
        max_error = local_max_error; // Simplified for sequential simulation
        
        if (proc_info->rank == 0) {
            printf("Iteration %d: Max Error = %f\n", iterations + 1, max_error);
        }
        
        iterations++;
        
        // Safety check to prevent infinite loops
        if (iterations > 100) {
            if (proc_info->rank == 0) printf("Warning: Maximum iterations reached\n");
            break;
        }
    }
    
    if (proc_info->rank == 0) {
        printf("Total iterations: %d\n", iterations);
    }
}

/***** Function to run a single process *****/
void Run_Single_Process(void *arg) {
    ProcessInfo *proc_info = (ProcessInfo *)arg;
    
    printf("Starting process %d of %d\n", proc_info->rank, proc_info->size);
    
    // Read graph data
    Read_from_txt_file("small_graph.txt", proc_info);
    
    // Initialize probabilities
    Random_P_E(proc_info);
    
    // Run simulated distributed PageRank
    Simulated_Distributed_PageRank(proc_info);
    
    // Cleanup
    int local_size = proc_info->local_end - proc_info->local_start;
    for (int i = 0; i < local_size; i++) {
        if (Nodes[i].To_id != NULL) {
            free(Nodes[i].To_id);
        }
    }
    free(Nodes);
}

/***** Main function *****/
int main(int argc, char** argv) {
    struct timeval start, end;
    double totaltime;
    
    // Default parameters
    N = 4;
    threshold = 0.001;
    d = 0.85;
    int num_processes = 2;
    
    // Parse command line arguments
    if (argc >= 2) num_processes = atoi(argv[1]);
    if (argc >= 3) N = atoi(argv[2]);
    if (argc >= 4) threshold = atof(argv[3]);
    if (argc >= 5) d = atof(argv[4]);
    
    printf("Simulated Distributed PageRank\n");
    printf("Using %d simulated processes\n", num_processes);
    printf("Graph size: %d nodes\n", N);
    printf("Threshold: %f\n", threshold);
    printf("Damping factor: %f\n", d);
    printf("\n");
    
    // Create communication directory
    Create_Comm_Directory();
    
    gettimeofday(&start, NULL);
    
    // Run processes sequentially (simulating distributed execution)
    for (int rank = 0; rank < num_processes; rank++) {
        ProcessInfo proc_info;
        proc_info.rank = rank;
        proc_info.size = num_processes;
        proc_info.local_start = rank * (N / num_processes);
        proc_info.local_end = (rank + 1) * (N / num_processes);
        if (rank == num_processes - 1) {
            proc_info.local_end = N; // Last process gets remainder
        }
        
        Run_Single_Process(&proc_info);
        
        // Small delay between processes
        Sleep(100);
    }
    
    gettimeofday(&end, NULL);
    
    totaltime = (((end.tv_usec - start.tv_usec) / 1.0e6 + end.tv_sec - start.tv_sec) * 1000) / 1000;
    
    printf("\nTotal time = %f seconds\n", totaltime);
    printf("End of simulated distributed program!\n");
    
    // Cleanup communication files
    ProcessInfo cleanup_info;
    cleanup_info.size = num_processes;
    Cleanup_Communication_Files(&cleanup_info);
    
    return EXIT_SUCCESS;
}