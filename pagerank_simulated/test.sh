#!/bin/bash

echo "Testing Simulated Distributed PageRank..."

# Check if executable exists
if [ ! -f ./pagerank_simulated ]; then
    echo "Error: pagerank_simulated not found. Please build first with ./build.sh"
    exit 1
fi

# Create test graph if it doesn't exist
if [ ! -f small_graph.txt ]; then
    echo "Creating small test graph..."
    cat > small_graph.txt << EOF
# Small test graph
0	1
0	2
1	2
2	3
3	0
3	1
EOF
fi

echo ""
echo "Running tests..."
echo ""

# Test 1: Basic test with 2 processes
echo "Test 1: Running with 2 processes..."
./pagerank_simulated 2 4 0.001
if [ $? -ne 0 ]; then
    echo "Test 1 failed!"
    exit 1
fi
echo "Test 1 passed!"
echo ""

# Test 2: Test with 4 processes
echo "Test 2: Running with 4 processes..."
./pagerank_simulated 4 4 0.001
if [ $? -ne 0 ]; then
    echo "Test 2 failed!"
    exit 1
fi
echo "Test 2 passed!"
echo ""

# Test 3: Test with custom parameters
echo "Test 3: Running with custom parameters..."
./pagerank_simulated 3 10 0.0001 0.9
if [ $? -ne 0 ]; then
    echo "Test 3 failed!"
    exit 1
fi
echo "Test 3 passed!"
echo ""

echo "All tests passed successfully!"