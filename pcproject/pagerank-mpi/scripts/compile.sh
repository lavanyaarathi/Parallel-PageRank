#!/bin/bash

# Compile the PageRank MPI project
mpicc -o pagerank-mpi src/main.c src/pagerank.c src/graph.c src/partition.c src/communication.c src/utils/timer.c -I include -lm