#ifndef COMMUNICATION_H
#define COMMUNICATION_H

#include <mpi.h>

// Function prototypes for communication routines
void send_rank_updates(double *local_ranks, int local_size, int dest_rank, MPI_Request *request);
void receive_rank_updates(double *local_ranks, int local_size, int source_rank, MPI_Request *request);
void exchange_rank_updates(double *local_ranks, int local_size, int num_processes);

#endif // COMMUNICATION_H