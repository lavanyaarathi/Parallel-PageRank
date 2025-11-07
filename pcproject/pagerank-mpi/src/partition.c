void partition_graph(int num_nodes, int num_processes, int *node_partition) {
    int nodes_per_process = num_nodes / num_processes;
    for (int i = 0; i < num_processes; i++) {
        for (int j = 0; j < nodes_per_process; j++) {
            node_partition[i * nodes_per_process + j] = i * nodes_per_process + j;
        }
    }
}

void print_partition(int num_processes, int *node_partition, int nodes_per_process) {
    for (int i = 0; i < num_processes; i++) {
        printf("Process %d: ", i);
        for (int j = 0; j < nodes_per_process; j++) {
            printf("%d ", node_partition[i * nodes_per_process + j]);
        }
        printf("\n");
    }
}