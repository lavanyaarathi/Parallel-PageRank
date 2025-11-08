Parallel Pagerank
=================

This repository contains three implementations of the PageRank algorithm in C:
1.  **Serial:** A standard, single-threaded implementation.
2.  **Pthreads:** A parallel implementation using POSIX threads.
3.  **MPI:** A distributed implementation using the Message Passing Interface (MPI).
4.  **Simulated Distributed:** A simulated distributed implementation that mimics the behavior of the MPI version without requiring an MPI installation.

How to Use
----------

### 1. Serial and Pthreads

These implementations are ideal for single-machine execution.

**Build:**
```bash
make
```

**Run Serial:**
```bash
./pagerank_serial <graph_file> <nodes> <threshold> <d>
```
**Example:**
```bash
./pagerank_serial web-Google.txt 916428 0.0001 0.85
```

**Run Pthreads:**
```bash
./pagerank_pthreads <graph_file> <nodes> <threshold> <d> <threads>
```
**Example:**
```bash
./pagerank_pthreads web-Google.txt 916428 0.0001 0.85 8
```

### 2. MPI (Distributed)

This implementation is designed for distributed memory systems and requires an MPI library (e.g., Open MPI, MPICH).

**Build:**
Navigate to the `pagerank_mpi` directory and use the provided build scripts:
*   **Windows:** `build.bat`
*   **Linux/macOS:** `build.sh`

**Run:**
Use the `mpiexec` command to run the compiled executable:
```bash
mpiexec -n <num_processes> ./pagerank_mpi <graph_file> <nodes> <threshold>
```
**Example:**
```bash
mpiexec -n 4 ./pagerank_mpi small_graph.txt 4 0.001
```

### 3. Simulated Distributed

This version simulates the distributed logic of the MPI implementation without the need for an MPI library, making it ideal for testing and development on a single machine.

**Build:**
Navigate to the `pagerank_simulated` directory and use the provided build scripts:
*   **Windows:** `build.bat`
*   **Linux/macOS:** `build.sh`

**Run:**
```bash
./pagerank_simulated <num_processes> <nodes> <threshold>
```
**Example:**
```bash
./pagerank_simulated 2 4 0.001
```

Output
------
Informational messages are printed to standard output.

[google-graph]: https://snap.stanford.edu/data/web-Google.html
