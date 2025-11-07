CC = mpicc
CFLAGS = -Iinclude -Wall -O2
SRC = src/main.c src/pagerank.c src/graph.c src/partition.c src/communication.c src/utils/timer.c
OBJ = $(SRC:.c=.o)
TARGET = pagerank_mpi

all: $(TARGET)

$(TARGET): $(OBJ)
	$(CC) -o $@ $^

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJ) $(TARGET)

run: $(TARGET)
	mpirun -np 4 ./$(TARGET) <input_file> <num_nodes> <threshold> <d> <num_threads>

.PHONY: all clean run