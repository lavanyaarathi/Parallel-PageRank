Sure, here's the contents for the file `/pagerank-mpi/pagerank-mpi/src/communication.c`:

#include "communication.h"
#include <mpi.h>
#include <stdlib.h>
#include <stdio.h>

void send_rank_updates(double *ranks, int count, int dest, MPI_Request *request) {
    MPI_Isend(ranks, count, MPI_DOUBLE, dest, 0, MPI_COMM_WORLD, request);
}

void receive_rank_updates(double *ranks, int count, int source, MPI_Request *request) {
    MPI_Irecv(ranks, count, MPI_DOUBLE, source, 0, MPI_COMM_WORLD, request);
}

void wait_for_requests(MPI_Request *request) {
    MPI_Wait(request, MPI_STATUS_IGNORE);
}

void exchange_rank_updates(double *local_ranks, int local_count, int num_processes) {
    MPI_Request *requests = malloc(num_processes * sizeof(MPI_Request));
    double *recv_buffer = malloc(local_count * sizeof(double));

    for (int i = 0; i < num_processes; i++) {
        if (i != MPI_COMM_WORLD) {
            send_rank_updates(local_ranks, local_count, i, &requests[i]);
            receive_rank_updates(recv_buffer, local_count, i, &requests[i]);
        }
    }

    for (int i = 0; i < num_processes; i++) {
        if (i != MPI_COMM_WORLD) {
            wait_for_requests(&requests[i]);
        }
    }

    free(requests);
    free(recv_buffer);
}