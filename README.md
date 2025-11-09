Parallel Pagerank
=================

This repository contains optimized implementations of the PageRank algorithm in C with comprehensive benchmarking, profiling, and visualization tools.

## 🚀 Quick Start - How to Run This Project

**New to the project?** Follow these steps:

### Windows Users

1. **Compile all implementations:**
   ```powershell
   .\build_all.bat
   ```

2. **Compare all three methods:**
   ```powershell
   python compare_methods.py pagerank_mpi\small_graph.txt 4 0.0001 0.85 4
   ```

3. **Or run individually:**
   ```powershell
   # Serial
   .\pagerank_serial\pagerank_serial.exe pagerank_mpi\small_graph.txt 4 0.0001 0.85
   
   # Pthreads (4 threads)
   .\pagerank_pthreads\pagerank_pthreads.exe pagerank_mpi\small_graph.txt 4 0.0001 0.85 4
   
   # MPI (4 processes)
   mpiexec -n 4 .\pagerank_mpi\pagerank_mpi.exe pagerank_mpi\small_graph.txt 4 0.0001 0.85
   ```

**Windows Setup Issues?** See [WINDOWS_SETUP.md](WINDOWS_SETUP.md)

**PowerShell Tips:** See [POWERSHELL_TIPS.md](POWERSHELL_TIPS.md) - Important: Use `.\` prefix to run executables!

**Want to use the Dashboard?** Run `.\run_dashboard.bat` (see below)

### Linux/macOS Users

1. **Compile all implementations:**
   ```bash
   cd pagerank_serial && make && cd ..
   cd pagerank_pthreads && make && cd ..
   cd pagerank_mpi && make && cd ..
   ```

2. **Compare all three methods:**
   ```bash
   python compare_methods.py pagerank_mpi/small_graph.txt 4 0.0001 0.85 4
   ```

3. **Or run individually:**
   ```bash
   # Serial
   ./pagerank_serial/pagerank_serial pagerank_mpi/small_graph.txt 4 0.0001 0.85
   
   # Pthreads (4 threads)
   ./pagerank_pthreads/pagerank_pthreads pagerank_mpi/small_graph.txt 4 0.0001 0.85 4
   
   # MPI (4 processes)
   mpiexec -n 4 ./pagerank_mpi/pagerank_mpi pagerank_mpi/small_graph.txt 4 0.0001 0.85
   ```

**📖 For detailed instructions:** See [QUICKSTART.md](QUICKSTART.md)

**🔍 To understand differences:** See [COMPARISON.md](COMPARISON.md) or [METHODS_OVERVIEW.md](METHODS_OVERVIEW.md)

## Implementations

1.  **Serial:** A standard, single-threaded implementation.
2.  **Pthreads:** A parallel implementation using POSIX threads.
3.  **MPI:** A distributed implementation using the Message Passing Interface (MPI) with CSR format and block decomposition.
4.  **Simulated Distributed:** A simulated distributed implementation that mimics the behavior of the MPI version without requiring an MPI installation.

## Key Features

- ✅ **Sparse Matrix Optimization**: CSR (Compressed Sparse Row) format for efficient memory usage
- ✅ **Block/Page Decomposition**: Reduced communication overhead in MPI
- ✅ **Adaptive Convergence**: L1 norm-based convergence detection
- ✅ **Benchmarking Suite**: Comprehensive performance analysis tools
- ✅ **Scalability Studies**: Strong and weak scaling analysis
- ✅ **Visualization Tools**: Graph visualization and convergence plots
- ✅ **Interactive Dashboard**: Web-based interface using Streamlit

See [FEATURES.md](FEATURES.md) for detailed documentation of all features.

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
mpiexec -n <num_processes> ./pagerank_mpi <graph_file> <nodes> <threshold> <d>
```
**Example:**
```bash
mpiexec -n 4 ./pagerank_mpi small_graph.txt 4 0.0001 0.85
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
All implementations output:
- Iteration-by-iteration convergence metrics (Max Error and L1 Norm)
- Total iterations
- Execution time
- Convergence status

Example output:
```
Iteration 1: Max Error = 0.123456, L1 Norm = 0.456789
Iteration 2: Max Error = 0.045678, L1 Norm = 0.123456
...
Converged based on L1 norm: 0.000095 < 0.0001
Total iterations: 15
Totaltime = 2.345 seconds
```

## Benchmarking & Analysis

### Run Benchmarks
```bash
python benchmark/benchmark.py graph.txt 1000 --threads 1 2 4 8
```

### Scalability Studies
```bash
python benchmark/scalability_study.py graph.txt 1000 --processes 1 2 4 8
```

### Visualization
```bash
python visualization/visualize_pagerank.py \
    --convergence output.txt \
    --graph graph.txt \
    --pagerank output.txt
```

### Interactive Dashboard

**Windows:**
```powershell
.\run_dashboard.bat
```

**Linux/macOS:**
```bash
cd dashboard
pip install -r requirements.txt
streamlit run app.py
```

The dashboard will open at http://localhost:8501

## Profiling

See `benchmark/profiling_tools.sh` for profiling commands using:
- **gprof**: Function-level profiling
- **perf**: Linux performance counters
- **mpiP**: MPI profiling
- **Valgrind**: Memory profiling

## Quick Comparison of Methods

| Method | Parallelization | Best For | Typical Speedup |
|--------|----------------|----------|-----------------|
| **Serial** | None | Small graphs, testing | 1x (baseline) |
| **Pthreads** | Threads (shared memory) | Multi-core machines | 2-8x |
| **MPI** | Processes (distributed) | Large graphs, clusters | 4-100x+ |

**Key Differences:**
- **Serial**: Simple, sequential processing
- **Pthreads**: Parallel on single machine using threads
- **MPI**: Distributed across multiple machines using message passing

See [COMPARISON.md](COMPARISON.md) for detailed explanation of how each method works.

## Quick Start

**New to the project?** Start here:
1. Read [QUICKSTART.md](QUICKSTART.md) for step-by-step instructions
2. Run the comparison script: `python compare_methods.py pagerank_mpi/small_graph.txt 4 0.0001 0.85 4`
3. See [COMPARISON.md](COMPARISON.md) to understand method differences

## Documentation

**Getting Started:**
- [WINDOWS_SETUP.md](WINDOWS_SETUP.md) - **Windows users start here!**
- [QUICKSTART.md](QUICKSTART.md) - Step-by-step getting started guide
- [METHODS_OVERVIEW.md](METHODS_OVERVIEW.md) - Visual overview of all methods
- [COMPARISON.md](COMPARISON.md) - Detailed technical comparison of all methods

**Features & Tools:**
- [FEATURES.md](FEATURES.md) - Detailed feature documentation
- [benchmark/README.md](benchmark/) - Benchmarking guide
- [dashboard/README.md](dashboard/README.md) - Dashboard setup

**Tools:**
- `compare_methods.py` - Quick comparison script (run all methods side-by-side)

[google-graph]: https://snap.stanford.edu/data/web-Google.html
