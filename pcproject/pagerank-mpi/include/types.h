#ifndef TYPES_H
#define TYPES_H

typedef struct {
    int id;          // Node ID
    double rank;    // PageRank value
    int con_size;   // Number of connections
    int from_size;  // Number of incoming connections
    int *From_id;   // Array of incoming connection IDs
} Node;

typedef struct {
    int num_nodes;  // Total number of nodes in the graph
    Node *nodes;    // Array of nodes
} Graph;

#endif // TYPES_H