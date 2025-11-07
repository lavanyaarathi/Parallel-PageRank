# MPI PageRank Implementation

This directory contains a distributed memory parallel implementation of the PageRank algorithm using MPI (Message Passing Interface).

## Features

- **Distributed Graph Partitioning**: The graph is partitioned across multiple MPI processes using block partitioning
- **Non-blocking Communication**: Uses MPI_Isend and MPI_Irecv for overlapping computation and communication
- **Scalable Design**: Can handle large graphs by distributing memory across multiple nodes
- **Load Balancing**: Automatic load balancing through block partitioning

## Implementation Details

### Graph Partitioning
- Each MPI process gets a contiguous range of nodes (block partitioning)
- Local adjacency lists are stored for the assigned nodes
- Rank updates are exchanged via MPI communication

### Non-blocking Communication
- Uses `MPI_Isend` and `MPI_Irecv` for asynchronous data exchange
- Overlaps computation and communication for better performance
- `MPI_Waitall` ensures all communications complete before proceeding

### Algorithm Flow
1. **Initialization**: Each process reads its portion of the graph
2. **Local Computation**: Compute local PageRank updates
3. **Communication**: Exchange rank values using non-blocking MPI calls
4. **Global Operations**: Compute global max error and sum using `MPI_Allreduce`
5. **Convergence Check**: Repeat until convergence threshold is met

## Compilation

### On Linux/Unix systems with make:
```bash
make
```

### On Windows:
```cmd
build.bat
```

### Manual compilation:
```bash
mpicc -Wall -O3 -std=c99 -c pagerank_mpi.c -o pagerank_mpi.o
mpicc pagerank_mpi.o -o pagerank_mpi -lm
```

## Usage

### Basic execution:
```bash
mpirun -np 4 ./pagerank_mpi graph_file.txt N threshold d
```

### Parameters:
- `graph_file.txt`: Graph file in edge list format (from_node\tto_node)
- `N`: Total number of nodes in the graph
- `threshold`: Convergence threshold (e.g., 0.0001)
- `d`: Damping factor (typically 0.85)

### Examples:
```bash
# Run with 2 processes
mpirun -np 2 ./pagerank_mpi web-Google.txt 1000 0.0001 0.85

# Run with 4 processes
mpirun -np 4 ./pagerank_mpi web-Google.txt 1000 0.0001 0.85

# Run with 8 processes
mpirun -np 8 ./pagerank_mpi web-Google.txt 1000 0.0001 0.85
```

## Testing

### Run tests with different process counts:
```bash
make test
```

This will run the algorithm with 2, 4, and 8 processes to verify scalability.

## Performance Considerations

1. **Process Count**: Optimal performance depends on graph size and system architecture
2. **Network**: Performance improves with high-speed interconnects (InfiniBand, etc.)
3. **Memory**: Each process only stores its partition, allowing larger graphs
4. **Communication Overlap**: Non-blocking operations help hide communication latency

## Output

The program outputs:
- Process distribution information
- Convergence progress (max error per iteration)
- Total number of iterations
- Total execution time

## Comparison with Other Versions

| Version | Memory Model | Communication | Scalability |
|---------|-------------|---------------|-------------|
| Serial | Single process | N/A | Limited by single node |
| Pthreads | Shared memory | Thread synchronization | Single node |
| MPI | Distributed memory | Message passing | Multi-node |

## Troubleshooting

1. **MPI not found**: Ensure MPI is installed and in your PATH
2. **Compilation errors**: Check that you have a C compiler and MPI development headers
3. **Runtime errors**: Verify the graph file exists and is in the correct format
4. **Performance issues**: Try different process counts and check network configuration