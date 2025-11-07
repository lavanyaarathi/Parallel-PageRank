#!/bin/bash
# Test script for MPI PageRank implementation

echo "MPI PageRank Test Script"
echo "========================"

# Check if mpirun is available
if ! command -v mpirun &> /dev/null; then
    echo "Error: mpirun not found. Please ensure MPI is installed."
    exit 1
fi

# Check if executable exists
if [ ! -f "pagerank_mpi" ]; then
    echo "Error: pagerank_mpi executable not found. Please compile first."
    exit 1
fi

# Create a small test graph if web-Google.txt doesn't exist
if [ ! -f "web-Google.txt" ]; then
    echo "Creating small test graph..."
    cat > small_graph.txt << EOF
# Small test graph
0	1
0	2
1	2
2	0
3	0
3	1
3	2
EOF
    GRAPH_FILE="small_graph.txt"
    NODES=4
else
    GRAPH_FILE="web-Google.txt"
    NODES=1000
fi

echo "Using graph file: $GRAPH_FILE"
echo "Number of nodes: $NODES"
echo ""

# Test with different process counts
for np in 2 4; do
    echo "Testing with $np processes..."
    echo "-------------------------"
    
    mpirun -np $np ./pagerank_mpi $GRAPH_FILE $NODES 0.001 0.85
    
    if [ $? -eq 0 ]; then
        echo "✓ Test with $np processes PASSED"
    else
        echo "✗ Test with $np processes FAILED"
    fi
    echo ""
done

echo "Test completed!"
echo ""
echo "For larger graphs, you can use:"
echo "  mpirun -np 8 ./pagerank_mpi web-Google.txt 10000 0.0001 0.85"