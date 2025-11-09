# Makefile for MPI PageRank Implementation
CC = mpicc
CFLAGS = -Wall -O3 -std=c99
LDFLAGS = -lm

# Target executable
TARGET = pagerank_mpi

# Source files for CSR implementation
SOURCES = pagerank_mpi.c csr_graph.c
HEADERS = pagerank_mpi.h csr_graph.h
OBJECTS = $(SOURCES:.c=.o)

# Default target
all: $(TARGET)

# Build the executable
$(TARGET): $(OBJECTS)
	$(CC) $(OBJECTS) -o $(TARGET) $(LDFLAGS)

# Compile source files
%.o: %.c $(HEADERS)
	$(CC) $(CFLAGS) -c $< -o $@

# Clean build artifacts
clean:
	rm -f $(OBJECTS) $(TARGET)

# Run with MPI (example with 4 processes)
run: $(TARGET)
	mpirun -np 4 ./$(TARGET) web-Google.txt 1000 0.0001 0.85

# Run with different process counts for testing
test: $(TARGET)
	@echo "Testing with 2 processes:"
	mpirun -np 2 ./$(TARGET) web-Google.txt 1000 0.0001 0.85
	@echo "\nTesting with 4 processes:"
	mpirun -np 4 ./$(TARGET) web-Google.txt 1000 0.0001 0.85
	@echo "\nTesting with 8 processes:"
	mpirun -np 8 ./$(TARGET) web-Google.txt 1000 0.0001 0.85

.PHONY: all clean run test