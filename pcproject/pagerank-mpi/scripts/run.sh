#!/bin/bash

# This script is used to execute the MPI application for the PageRank algorithm.

# Check if the number of processes is provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <number_of_processes> <graph_file>"
    exit 1
fi

# Assign command-line arguments to variables
NUM_PROCESSES=$1
GRAPH_FILE=$2

# Execute the MPI application
mpirun -np $NUM_PROCESSES ./pagerank_mpi $GRAPH_FILE