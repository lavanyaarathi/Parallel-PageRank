# PageRank MPI Project

This project implements the PageRank algorithm using MPI (Message Passing Interface) for distributed memory parallelism. The implementation includes graph partitioning, local adjacency lists, and non-blocking MPI routines for communication between processes.

## Project Structure

- **src/**: Contains the source code for the application.
  - **main.c**: Entry point of the MPI application.
  - **pagerank.c**: Implementation of the PageRank algorithm.
  - **pagerank.h**: Header file for PageRank functions.
  - **graph.c**: Functions for reading and managing the graph structure.
  - **graph.h**: Header file for graph functions.
  - **partition.c**: Functions for partitioning the graph across processes.
  - **partition.h**: Header file for partitioning functions.
  - **communication.c**: MPI communication routines.
  - **communication.h**: Header file for communication functions.
  - **utils/**: Utility functions, including timers.
    - **timer.c**: Timer utility functions.
    - **timer.h**: Header file for timer functions.

- **include/**: Contains header files for custom types and structures.
  - **types.h**: Defines custom types used throughout the project.

- **scripts/**: Contains scripts for running and compiling the application.
  - **run.sh**: Script to execute the MPI application.
  - **compile.sh**: Script to compile the project.

- **Makefile**: Build instructions for the project.

- **test/**: Contains unit tests for the application.
  - **test_graph.c**: Unit tests for graph functions.
  - **test_pagerank.c**: Unit tests for PageRank functions.

## Building the Project

To build the project, run the following command in the terminal:

```bash
make
```

This will compile the source files and create the executable.

## Running the Application

To run the MPI application, use the following command:

```bash
mpirun -np <number_of_processes> ./pagerank_mpi <graph_file> <num_nodes> <threshold> <d>
```

Replace `<number_of_processes>`, `<graph_file>`, `<num_nodes>`, `<threshold>`, and `<d>` with appropriate values.

## Overview of the PageRank Algorithm

The PageRank algorithm is a method for ranking web pages in search engine results. It works by counting the number and quality of links to a page to determine a rough estimate of the website's importance. The algorithm is based on the premise that more important websites are likely to receive more links from other websites.

This implementation uses MPI to distribute the computation across multiple processes, allowing for efficient handling of large graphs. The graph is partitioned among the processes, and each process computes the PageRank for its local subset of nodes, communicating updates with neighboring processes as needed.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.