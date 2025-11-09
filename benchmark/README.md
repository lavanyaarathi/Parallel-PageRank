# Benchmarking and Profiling Tools

This directory contains tools for benchmarking, profiling, and analyzing PageRank implementations.

## Benchmarking Suite

### benchmark.py
Comprehensive benchmarking tool that compares all implementations.

**Usage**:
```bash
python benchmark.py graph.txt 1000 --threads 1 2 4 8 --threshold 0.0001 --d 0.85
```

**Output**:
- JSON file with detailed results
- CSV file for easy analysis
- Console summary with speedup and efficiency metrics

### scalability_study.py
Performs strong and weak scaling studies.

**Usage**:
```bash
# Strong scaling (fixed problem size)
python scalability_study.py graph.txt 1000 --strong-only

# Weak scaling (proportional problem size)
python scalability_study.py graph.txt 1000 --weak-only

# Both
python scalability_study.py graph.txt 1000
```

**Output**:
- JSON results file
- PNG plots for scalability analysis

## Profiling

### profiling_tools.sh
Guide and commands for various profiling tools.

**Available Tools**:
- **gprof**: GNU Profiler for serial/pthreads
- **perf**: Linux performance counters
- **mpiP**: MPI-specific profiler
- **Valgrind**: Memory and call profiling
- **Intel VTune**: Advanced performance analysis

**Usage**:
```bash
bash profiling_tools.sh
```

## Example Workflow

1. **Run benchmarks**:
   ```bash
   python benchmark.py graph.txt 1000
   ```

2. **Analyze scalability**:
   ```bash
   python scalability_study.py graph.txt 1000 --processes 1 2 4 8 16
   ```

3. **Profile with gprof**:
   ```bash
   gcc -pg -O3 pagerank_serial/pagerank_serial.c -lm -o pagerank_serial/pagerank_serial
   ./pagerank_serial/pagerank_serial graph.txt 1000 0.0001 0.85
   gprof pagerank_serial/pagerank_serial gmon.out > analysis.txt
   ```

4. **Profile with perf**:
   ```bash
   perf record ./pagerank_serial/pagerank_serial graph.txt 1000 0.0001 0.85
   perf report
   ```

