#!/bin/bash
# Profiling tools setup and usage guide for PageRank implementations

# This script provides commands for profiling with gprof, perf, and mpiP

echo "PageRank Profiling Tools"
echo "========================"
echo ""

# Check if tools are available
check_tool() {
    if command -v $1 &> /dev/null; then
        echo "✓ $1 is available"
        return 0
    else
        echo "✗ $1 is not available"
        return 1
    fi
}

echo "Checking for profiling tools:"
check_tool gprof
check_tool perf
check_tool mpicc

echo ""
echo "Profiling Commands:"
echo "==================="
echo ""

echo "1. gprof (GNU Profiler) - Serial and Pthreads:"
echo "   Compile with -pg flag:"
echo "   gcc -pg -O3 pagerank_serial.c -lm -o pagerank_serial"
echo "   ./pagerank_serial graph.txt 1000 0.0001 0.85"
echo "   gprof pagerank_serial gmon.out > analysis.txt"
echo ""

echo "2. perf (Linux Performance Counter):"
echo "   perf record ./pagerank_serial graph.txt 1000 0.0001 0.85"
echo "   perf report"
echo "   perf stat ./pagerank_serial graph.txt 1000 0.0001 0.85"
echo ""

echo "3. mpiP (MPI Profiler):"
echo "   Compile MPI with mpiP:"
echo "   mpicc -g -O3 -I/path/to/mpiP/include -L/path/to/mpiP/lib -lmpiP pagerank_mpi.c -o pagerank_mpi"
echo "   mpiexec -n 4 ./pagerank_mpi graph.txt 1000 0.0001 0.85"
echo "   (generates mpiP output file)"
echo ""

echo "4. Valgrind (Memory profiling):"
echo "   valgrind --tool=callgrind ./pagerank_serial graph.txt 1000 0.0001 0.85"
echo "   kcachegrind callgrind.out.*"
echo ""

echo "5. Intel VTune (if available):"
echo "   vtune -collect hotspots -result-dir ./vtune_results ./pagerank_serial graph.txt 1000 0.0001 0.85"
echo ""

