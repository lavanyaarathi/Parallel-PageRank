# PageRank Implementation Features

This document describes all the features and optimizations implemented in the PageRank project.

## ✅ Bug Fixes

### Fixed Issues
1. **Assert statements**: Fixed incorrect assignment (`=`) to comparison (`==`) in assert statements
2. **Hardcoded filename**: Fixed hardcoded "web-Google.txt" in pthreads version to use command-line argument
3. **Duplicate declarations**: Removed duplicate `num_threads` declaration in pthreads version
4. **CSR comment handling**: Added support for comment lines (starting with `#`) in graph files

## 🚀 Optimizations

### 1. Sparse Matrix Optimization (CSR Format)
- **Location**: `pagerank_mpi/csr_graph.c`, `pagerank_mpi/csr_graph.h`
- **Benefits**:
  - Reduced memory usage for sparse graphs
  - Faster matrix-vector multiplication
  - Efficient storage of large graphs
- **Implementation**: Compressed Sparse Row (CSR) format with:
  - `row_ptr`: Row pointers array
  - `col_ind`: Column indices array
  - Optimized graph reading with comment support

### 2. Block/Page Decomposition
- **Location**: `pagerank_mpi/pagerank_mpi.c` - `Distributed_PageRank_Block()`
- **Benefits**:
  - Reduced communication overhead in MPI
  - Better load balancing
  - Periodic synchronization instead of every iteration
- **How it works**:
  - Divides graph into blocks assigned to each process
  - Performs multiple local iterations before synchronization
  - Synchronizes periodically to update global state

### 3. Adaptive Convergence
- **Location**: All implementations (serial, pthreads, MPI)
- **Benefits**:
  - More robust convergence detection
  - Stops when L1 norm < threshold (instead of just max error)
  - Faster convergence for well-conditioned graphs
- **Implementation**:
  - Computes both max error and L1 norm each iteration
  - Uses L1 norm as primary convergence criterion
  - Falls back to max error if needed

### 4. Optimized MPI Communication
- **Location**: `pagerank_mpi/pagerank_mpi.c`
- **Improvements**:
  - Uses `MPI_Allgatherv` for efficient data gathering
  - Parallel computation of dangling node sums
  - Optimized CSR matrix-vector multiplication
  - Reduced redundant broadcasts

## 📊 Performance Analysis Tools

### Benchmarking Suite
- **Location**: `benchmark/benchmark.py`
- **Features**:
  - Compares all implementations (Serial, Pthreads, MPI)
  - Measures execution time, speedup, and efficiency
  - Exports results to JSON and CSV
  - Supports custom thread/process counts

**Usage**:
```bash
python benchmark/benchmark.py graph.txt 1000 --threads 1 2 4 8
```

### Scalability Studies
- **Location**: `benchmark/scalability_study.py`
- **Features**:
  - **Strong Scaling**: Fixed problem size, increasing processors
  - **Weak Scaling**: Problem size increases with processors
  - Automatic plot generation
  - Efficiency calculations

**Usage**:
```bash
python benchmark/scalability_study.py graph.txt 1000 --processes 1 2 4 8
```

### Profiling Tools
- **Location**: `benchmark/profiling_tools.sh`
- **Supported Tools**:
  - **gprof**: Function-level profiling for serial/pthreads
  - **perf**: Linux performance counters
  - **mpiP**: MPI-specific profiling
  - **Valgrind**: Memory and call profiling
  - **Intel VTune**: Advanced performance analysis

**Usage**:
```bash
bash benchmark/profiling_tools.sh
```

## 📈 Visualization Tools

### Graph Visualization
- **Location**: `visualization/visualize_pagerank.py`
- **Features**:
  - Convergence plots (max error and L1 norm)
  - Graph structure visualization with NetworkX
  - Top-ranked nodes bar charts
  - PageRank value distribution histograms

**Usage**:
```bash
python visualization/visualize_pagerank.py \
    --convergence output.txt \
    --graph graph.txt \
    --pagerank output.txt \
    --top-n 20
```

### Interactive Dashboard
- **Location**: `dashboard/app.py`
- **Technology**: Streamlit
- **Features**:
  - Web-based interface
  - Upload graph files or use samples
  - Run any implementation (Serial/Pthreads/MPI)
  - Real-time convergence visualization
  - Top-ranked nodes display
  - PageRank distribution analysis
  - Graph structure visualization

**Usage**:
```bash
cd dashboard
pip install -r requirements.txt
streamlit run app.py
```

## 🔧 Build Options

### Standard Build
```bash
# Serial
cd pagerank_serial && make

# Pthreads
cd pagerank_pthreads && make

# MPI
cd pagerank_mpi && make
```

### Profiling Build (gprof)
```bash
# Serial with profiling
gcc -pg -O3 pagerank_serial/pagerank_serial.c -lm -o pagerank_serial/pagerank_serial

# Pthreads with profiling
gcc -pg -O3 pagerank_pthreads/pagerank_pthreads.c -lpthread -lm -o pagerank_pthreads/pagerank_pthreads
```

## 📝 Output Format

All implementations now output:
- Iteration number
- Max error per iteration
- L1 norm per iteration
- Total iterations
- Execution time
- Convergence status (L1 norm or max error)

Example output:
```
Iteration 1: Max Error = 0.123456, L1 Norm = 0.456789
Iteration 2: Max Error = 0.045678, L1 Norm = 0.123456
...
Converged based on L1 norm: 0.000095 < 0.0001
Total iterations: 15
Totaltime = 2.345 seconds
```

## 🎯 Performance Metrics

The benchmarking tools measure:
- **Execution Time**: Wall-clock time for algorithm completion
- **Speedup**: Serial time / Parallel time
- **Efficiency**: Speedup / Number of processors
- **Scalability**: How performance scales with problem/processor size

## 📚 Additional Resources

- See `README.md` for basic usage
- See `benchmark/README.md` for benchmarking details
- See `dashboard/README.md` for dashboard setup

