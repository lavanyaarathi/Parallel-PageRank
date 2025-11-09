# Quick Start Guide

This guide will help you get started with the PageRank project quickly.

## Prerequisites

### Required
- **C Compiler**: GCC or Clang
  - **Windows**: MinGW-w64 or MSYS2 (see [WINDOWS_SETUP.md](WINDOWS_SETUP.md))
  - **Linux**: `sudo apt-get install build-essential`
  - **macOS**: Xcode Command Line Tools
- **Python 3.8+**: For benchmarking and visualization (optional)

### Optional (for specific features)
- **MPI**: 
  - **Windows**: Microsoft MPI (see [WINDOWS_SETUP.md](WINDOWS_SETUP.md))
  - **Linux**: `sudo apt-get install openmpi-bin libopenmpi-dev`
  - **macOS**: `brew install open-mpi`
- **pthreads**: Usually included with GCC
- **Python packages**: See `dashboard/requirements.txt` for dashboard

**Windows Users**: See [WINDOWS_SETUP.md](WINDOWS_SETUP.md) for detailed Windows-specific instructions.

## Step 1: Compile the Implementations

### Windows Users

**Quick Build (All at once):**
```powershell
.\build_all.bat
```

**Or build individually:**
```powershell
cd pagerank_serial
.\build.bat
cd ..

cd pagerank_pthreads
.\build.bat
cd ..

cd pagerank_mpi
.\build.bat
cd ..
```

This creates `.exe` files in each directory.

**Note**: If you get errors, see [WINDOWS_SETUP.md](WINDOWS_SETUP.md) for troubleshooting.

### Linux/macOS Users

**Serial Implementation:**
```bash
cd pagerank_serial
make
cd ..
```

**Pthreads Implementation:**
```bash
cd pagerank_pthreads
make
cd ..
```

**MPI Implementation:**
```bash
cd pagerank_mpi
make
cd ..
```

This creates executables in each directory.

**Note**: MPI requires MPI runtime. Install with:
- Ubuntu/Debian: `sudo apt-get install openmpi-bin openmpi-common libopenmpi-dev`
- macOS: `brew install open-mpi`
- Windows: Install MS-MPI from Microsoft

## Step 2: Prepare a Graph File

Graph files should be in the following format:
```
# Comments start with #
# Format: from_node to_node
0	1
0	2
1	2
2	0
3	0
```

Or with spaces:
```
0 1
0 2
1 2
2 0
```

**Sample graph**: `pagerank_mpi/small_graph.txt` (4 nodes, 7 edges)

## Step 3: Run the Implementations

### Run Serial Version

**Windows:**
```powershell
.\pagerank_serial\pagerank_serial.exe <graph_file> <nodes> <threshold> <d>
```

**Linux/macOS:**
```bash
./pagerank_serial/pagerank_serial <graph_file> <nodes> <threshold> <d>
```

**Example:**
```powershell
# Windows
.\pagerank_serial\pagerank_serial.exe pagerank_mpi\small_graph.txt 4 0.0001 0.85

# Linux/macOS
./pagerank_serial/pagerank_serial pagerank_mpi/small_graph.txt 4 0.0001 0.85
```

**Parameters:**
- `graph_file`: Path to graph file
- `nodes`: Number of nodes in graph
- `threshold`: Convergence threshold (e.g., 0.0001)
- `d`: Damping factor (typically 0.85)

### Run Pthreads Version

**Windows:**
```powershell
.\pagerank_pthreads\pagerank_pthreads.exe <graph_file> <nodes> <threshold> <d> <threads>
```

**Linux/macOS:**
```bash
./pagerank_pthreads/pagerank_pthreads <graph_file> <nodes> <threshold> <d> <threads>
```

**Example:**
```powershell
# Windows
.\pagerank_pthreads\pagerank_pthreads.exe pagerank_mpi\small_graph.txt 4 0.0001 0.85 4

# Linux/macOS
./pagerank_pthreads/pagerank_pthreads pagerank_mpi/small_graph.txt 4 0.0001 0.85 4
```

**Additional parameter:**
- `threads`: Number of threads to use (1-64)

### Run MPI Version

**Windows:**
```powershell
mpiexec -n <num_processes> .\pagerank_mpi\pagerank_mpi.exe <graph_file> <nodes> <threshold> <d>
```

**Linux/macOS:**
```bash
mpiexec -n <num_processes> ./pagerank_mpi/pagerank_mpi <graph_file> <nodes> <threshold> <d>
```

**Example:**
```powershell
# Windows
mpiexec -n 4 .\pagerank_mpi\pagerank_mpi.exe pagerank_mpi\small_graph.txt 4 0.0001 0.85

# Linux/macOS
mpiexec -n 4 ./pagerank_mpi/pagerank_mpi pagerank_mpi/small_graph.txt 4 0.0001 0.85
```

**Note**: On Windows, use `mpiexec` (not `mpirun`). Make sure MPI is in your PATH.

**Additional parameter:**
- `num_processes`: Number of MPI processes (via `-n` flag)

## Step 4: Understand the Output

All implementations output similar information:

```
Iteration 1: Max Error = 0.123456, L1 Norm = 0.456789
Iteration 2: Max Error = 0.045678, L1 Norm = 0.123456
Iteration 3: Max Error = 0.012345, L1 Norm = 0.034567
...
Converged based on L1 norm: 0.000095 < 0.0001
Total iterations: 15
Totaltime = 2.345 seconds
```

**What it means:**
- **Max Error**: Maximum change in PageRank value for any single node
- **L1 Norm**: Sum of absolute changes across all nodes (more robust)
- **Converged**: Algorithm stopped when L1 norm < threshold
- **Total iterations**: Number of iterations until convergence
- **Totaltime**: Total execution time in seconds

## Step 5: Compare Methods (Optional)

### Quick Comparison Script

Create a simple comparison:

```bash
# Run all three methods
echo "=== Serial ===" > comparison.txt
./pagerank_serial/pagerank_serial pagerank_mpi/small_graph.txt 4 0.0001 0.85 >> comparison.txt

echo "=== Pthreads (4 threads) ===" >> comparison.txt
./pagerank_pthreads/pagerank_pthreads pagerank_mpi/small_graph.txt 4 0.0001 0.85 4 >> comparison.txt

echo "=== MPI (4 processes) ===" >> comparison.txt
mpiexec -n 4 ./pagerank_mpi/pagerank_mpi pagerank_mpi/small_graph.txt 4 0.0001 0.85 >> comparison.txt

cat comparison.txt
```

### Automated Benchmarking

```bash
python benchmark/benchmark.py pagerank_mpi/small_graph.txt 4 --threads 1 2 4
```

This will:
- Run Serial, Pthreads (1,2,4 threads), and MPI (1,2,4 processes)
- Compare execution times
- Calculate speedup and efficiency
- Save results to `benchmark_results.json` and `benchmark_results.csv`

## Step 6: Visualize Results (Optional)

### Using Python Visualization

```bash
# First, save output to a file
./pagerank_serial/pagerank_serial pagerank_mpi/small_graph.txt 4 0.0001 0.85 > output.txt

# Visualize
python visualization/visualize_pagerank.py \
    --convergence output.txt \
    --graph pagerank_mpi/small_graph.txt \
    --pagerank output.txt \
    --top-n 10
```

This creates plots in the `plots/` directory:
- `convergence.png`: Convergence over iterations
- `graph_visualization.png`: Graph structure
- `top_ranked.png`: Top-ranked nodes
- `rank_distribution.png`: PageRank distribution

### Using Interactive Dashboard

```bash
cd dashboard
pip install -r requirements.txt
streamlit run app.py
```

Then:
1. Open browser to `http://localhost:8501`
2. Upload a graph file or use sample
3. Select implementation and parameters
4. Click "Run PageRank"
5. View results and visualizations

## Common Issues and Solutions

### Issue: "Error opening data file"
**Solution**: Check that the graph file path is correct and the file exists.

### Issue: "mpiexec: command not found"
**Solution**: Install MPI runtime (see Prerequisites section).

### Issue: "pthreads not found"
**Solution**: Install build-essential (Linux) or Xcode command line tools (macOS).

### Issue: Results differ between implementations
**Solution**: This is normal! Small floating-point differences are expected. The final PageRank values should be very similar (within 0.001).

### Issue: MPI version is slower than Serial
**Solution**: For small graphs, MPI overhead dominates. Use MPI only for large graphs (> 10K nodes) or many processes.

## Testing with Different Graph Sizes

### Small Graph (Testing)
```bash
# 4 nodes, 7 edges
./pagerank_serial/pagerank_serial pagerank_mpi/small_graph.txt 4 0.0001 0.85
```

### Medium Graph (Performance Testing)
Download a medium graph (e.g., from SNAP dataset):
```bash
# Example: 1000 nodes
./pagerank_serial/pagerank_serial medium_graph.txt 1000 0.0001 0.85
./pagerank_pthreads/pagerank_pthreads medium_graph.txt 1000 0.0001 0.85 8
mpiexec -n 8 ./pagerank_mpi/pagerank_mpi medium_graph.txt 1000 0.0001 0.85
```

### Large Graph (Scalability Testing)
```bash
# Example: 100K+ nodes
# Use MPI for best performance
mpiexec -n 16 ./pagerank_mpi/pagerank_mpi large_graph.txt 100000 0.0001 0.85
```

## Next Steps

1. **Read [COMPARISON.md](COMPARISON.md)**: Understand differences between methods
2. **Read [FEATURES.md](FEATURES.md)**: Learn about all features
3. **Run benchmarks**: Compare performance across methods
4. **Try visualization**: See convergence and graph structure
5. **Experiment**: Try different parameters (threshold, damping factor, threads)

## Getting Help

- Check `README.md` for basic usage
- See `FEATURES.md` for feature documentation
- Review `COMPARISON.md` for method differences
- Check code comments for implementation details

## Example Workflow

Here's a complete example workflow:

```bash
# 1. Compile all implementations
cd pagerank_serial && make && cd ..
cd pagerank_pthreads && make && cd ..
cd pagerank_mpi && make && cd ..

# 2. Test with small graph
./pagerank_serial/pagerank_serial pagerank_mpi/small_graph.txt 4 0.0001 0.85

# 3. Compare methods
python benchmark/benchmark.py pagerank_mpi/small_graph.txt 4 --threads 1 2 4

# 4. Visualize results
./pagerank_serial/pagerank_serial pagerank_mpi/small_graph.txt 4 0.0001 0.85 > output.txt
python visualization/visualize_pagerank.py --convergence output.txt --graph pagerank_mpi/small_graph.txt --pagerank output.txt

# 5. Use dashboard
cd dashboard && streamlit run app.py
```

Happy computing! 🚀

