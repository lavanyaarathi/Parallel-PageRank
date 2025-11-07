void start_timer(double *start) {
    *start = MPI_Wtime();
}

void stop_timer(double *start, double *end) {
    *end = MPI_Wtime();
}

double elapsed_time(double start, double end) {
    return end - start;
}