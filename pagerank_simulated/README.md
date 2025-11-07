# Simulated Distributed PageRank

This is a simulated distributed implementation of the PageRank algorithm that **does not require MPI installation**. It simulates distributed behavior using file-based communication between processes that run sequentially.

## Features

- **No MPI Dependency**: Uses standard C libraries only
- **File-based Communication**: Simulates inter-process communication using files
- **Sequential Simulation**: Runs processes one after another (simulates parallel execution)
- **Cross-platform**: Works on Windows and Unix-like systems
- **Easy to Build**: Uses standard C compilers (gcc, clang, or Visual Studio)

## How It Works

1. **Simulated Processes**: Instead of true parallel processes, this implementation runs processes sequentially
2. **File-based Communication**: Processes communicate by reading/writing files in a `comm_files` directory
3. **Data Partitioning**: Graph data is partitioned across simulated processes
4. **Synchronization**: Uses file existence and timing delays to simulate process synchronization

## Building

### Windows
```cmd
build.bat
```

### Linux/Unix
```bash
chmod +x build.sh
./build.sh
```

## Usage

Basic usage:
```bash
./pagerank_simulated [num_processes] [num_nodes] [threshold] [damping_factor]
```

Examples:
```bash
# Run with 2 processes (default)
./pagerank_simulated

# Run with 4 processes
./pagerank_simulated 4

# Run with 4 processes, 10 nodes, threshold 0.001
./pagerank_simulated 4 10 0.001

# Run with 4 processes, 10 nodes, threshold 0.001, damping 0.9
./pagerank_simulated 4 10 0.001 0.9
```

## Testing

### Windows
```cmd
test.bat
```

### Linux/Unix
```bash
chmod +x test.sh
./test.sh
```

## Implementation Details

### Communication Files
The implementation creates several types of files in the `comm_files` directory:
- `rank_X_data.txt`: Contains graph data for process X
- `rank_X_ranks.txt`: Contains rank values from process X
- `rank_X_max_error.txt`: Contains maximum error from process X
- `rank_X_sum.txt`: Contains sum of dangling node contributions from process X

### Process Simulation
- Processes run sequentially, not in parallel
- File-based communication simulates message passing
- Timing delays simulate synchronization overhead
- Each process handles a partition of the graph

### Algorithm Flow
1. **Initialization**: Create communication directory and initialize data structures
2. **Data Distribution**: Rank 0 reads the graph file and distributes data to other "processes"
3. **Local Computation**: Each process computes PageRank on its partition
4. **Communication**: Exchange rank values using files
5. **Global Operations**: Compute global max error and sums using file-based reduction
6. **Iteration**: Repeat until convergence or max iterations
7. **Cleanup**: Remove communication files

## Limitations

- **Sequential Execution**: Not truly parallel, runs processes one at a time
- **File I/O Overhead**: Communication through files is slower than memory
- **No Real Concurrency**: Cannot exploit true parallel processing
- **Timing Delays**: Artificial delays for synchronization

## Comparison with MPI Version

| Feature | Simulated Version | MPI Version |
|---------|------------------|-------------|
| Parallel Execution | ❌ Sequential | ✅ True Parallel |
| MPI Required | ❌ No | ✅ Yes |
| Communication | File-based | Message Passing |
| Performance | Slower | Faster |
| Setup Complexity | Simple | Complex |
| Learning Value | High | Higher |

## When to Use This Version

- When MPI is not available or cannot be installed
- For educational purposes to understand distributed algorithms
- For development and debugging before MPI implementation
- For systems with strict software installation restrictions
- For quick prototyping and testing

## Graph File Format

The implementation expects graph files in the same format as the original:
```
# Comments start with #
from_node_id\tto_node_id
0\t1
0\t2
1\t2
2\t3
```

Default test file is `small_graph.txt` with 4 nodes and 6 edges.