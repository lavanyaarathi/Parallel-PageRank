# PageRank Implementation Comparison

This document explains the differences between the various PageRank calculation methods and their trade-offs.

## Overview of Methods

The project implements PageRank using three different parallelization strategies:

1. **Serial** - Single-threaded baseline
2. **Pthreads** - Shared-memory parallelism
3. **MPI** - Distributed-memory parallelism

## Method Comparison

### 1. Serial Implementation

**How it works:**
- Single-threaded execution
- Processes all nodes sequentially
- No parallelization overhead
- Simple and easy to understand

**Algorithm Flow:**
```
1. Read graph from file
2. Initialize PageRank values uniformly (1/N)
3. For each iteration:
   a. Update old PageRank values
   b. For each node, compute new PageRank from neighbors
   c. Handle dangling nodes (nodes with no outgoing links)
   d. Apply damping factor
   e. Check convergence (L1 norm or max error)
4. Output final PageRank values
```

**Characteristics:**
- ✅ Simple implementation
- ✅ No synchronization overhead
- ✅ Easy to debug
- ❌ No parallelization
- ❌ Slower for large graphs
- ❌ Limited to single machine

**Best for:**
- Small to medium graphs (< 100K nodes)
- Development and testing
- Baseline performance comparison

---

### 2. Pthreads Implementation (Shared Memory)

**How it works:**
- Uses POSIX threads for parallelization
- All threads share the same memory space
- Divides nodes among threads
- Uses mutex locks for thread-safe updates

**Algorithm Flow:**
```
1. Read graph from file
2. Initialize PageRank values uniformly (1/N)
3. Create N threads, each assigned a subset of nodes
4. For each iteration:
   a. Threads reinitialize their assigned nodes (parallel)
   b. Threads compute PageRank contributions (parallel)
      - Each thread processes its assigned nodes
      - Uses mutex locks for shared variables (sum, max_error)
   c. Threads compute local max error and L1 norm (parallel)
   d. Synchronize threads (barrier)
   e. Check convergence
5. Output final PageRank values
```

**Key Differences from Serial:**
- **Data Structure**: Uses `From_id` array (incoming links) instead of `To_id` (outgoing links)
  - This allows each thread to update its assigned nodes independently
  - Reduces contention on shared data
- **Thread Safety**: Uses mutex locks for:
  - Global sum accumulation (dangling nodes)
  - Max error calculation
  - L1 norm accumulation
- **Work Distribution**: Nodes are divided evenly among threads

**Characteristics:**
- ✅ Good speedup on multi-core machines
- ✅ Shared memory (no data copying)
- ✅ Relatively simple parallelization
- ❌ Limited by number of CPU cores
- ❌ Mutex overhead for synchronization
- ❌ Memory contention on large graphs
- ❌ Still limited to single machine

**Best for:**
- Multi-core single machines
- Medium to large graphs (100K - 10M nodes)
- When you have 4-16 CPU cores available

**Thread Safety Mechanisms:**
```c
pthread_mutex_t locksum;  // Protects global sum variable
pthread_mutex_t lockmax;  // Protects max error calculation
pthread_mutex_t lockl1;   // Protects L1 norm accumulation
```

---

### 3. MPI Implementation (Distributed Memory)

**How it works:**
- Uses Message Passing Interface for distributed computing
- Each process has its own memory space
- Graph is partitioned across processes
- Processes communicate via MPI messages
- Uses CSR (Compressed Sparse Row) format for efficiency

**Algorithm Flow:**
```
1. Process 0 reads graph and converts to CSR format
2. Broadcast CSR graph to all processes
3. Each process gets a partition of nodes
4. For each iteration:
   a. Gather all PageRank values from all processes (MPI_Allgatherv)
   b. Each process computes dangling node sum for its partition
   c. Reduce dangling sums across all processes (MPI_Allreduce)
   d. Each process computes PageRank for its assigned nodes
      - Uses CSR format for efficient matrix-vector multiplication
      - Only processes nodes in its partition
   e. Each process computes local max error and L1 norm
   f. Reduce max error and L1 norm across processes (MPI_Allreduce)
   g. Check convergence
5. Output final PageRank values
```

**Key Differences from Serial/Pthreads:**

1. **CSR Format (Compressed Sparse Row)**:
   ```c
   typedef struct {
       int num_nodes;
       int num_edges;
       int *row_ptr;    // Row pointers (size = num_nodes + 1)
       int *col_ind;    // Column indices (size = num_edges)
   } CSRGraph;
   ```
   - Stores only non-zero entries
   - Efficient matrix-vector multiplication
   - Reduced memory footprint

2. **Data Partitioning**:
   - Nodes are divided among processes
   - Each process only stores its partition
   - Must communicate to get values from other partitions

3. **Communication Patterns**:
   - `MPI_Allgatherv`: Gather all PageRank values to all processes
   - `MPI_Allreduce`: Compute global sums/maxima
   - `MPI_Barrier`: Synchronize all processes

4. **Block Decomposition Option**:
   - Can perform multiple local iterations before synchronization
   - Reduces communication frequency
   - Better for high-latency networks

**Characteristics:**
- ✅ Can scale to many machines (100s-1000s of processes)
- ✅ Can handle very large graphs (millions to billions of nodes)
- ✅ Efficient memory usage with CSR format
- ✅ No shared memory limitations
- ❌ Communication overhead
- ❌ More complex implementation
- ❌ Requires MPI runtime
- ❌ Network latency affects performance

**Best for:**
- Very large graphs (> 10M nodes)
- Distributed computing clusters
- When you need to scale beyond single machine
- Research and production systems

---

## Performance Comparison

### Theoretical Complexity

All methods have the same algorithmic complexity:
- **Time**: O(I × (N + E)) where I = iterations, N = nodes, E = edges
- **Space**: O(N + E)

### Practical Performance

| Method | Speedup | Scalability | Memory | Communication |
|--------|---------|-------------|--------|---------------|
| Serial | 1x (baseline) | Single core | O(N+E) | None |
| Pthreads | 2-8x (typical) | Up to CPU cores | O(N+E) | Shared memory |
| MPI | 4-100x+ | Up to cluster size | O(N/P + E/P) | Network messages |

### Speedup Factors

**Pthreads:**
- Ideal speedup: N (number of threads)
- Actual speedup: 0.6-0.8 × N (due to synchronization overhead)
- Example: 8 threads → ~5-6x speedup

**MPI:**
- Ideal speedup: P (number of processes)
- Actual speedup: 0.4-0.7 × P (due to communication overhead)
- Example: 16 processes → ~6-11x speedup
- Better speedup on larger problems (communication overhead is amortized)

---

## Memory Usage Comparison

### Serial
```
Memory = N × (sizeof(Node)) + E × sizeof(int)
Node structure: ~40 bytes (p_t0, p_t1, e, To_id array, con_size)
Total: ~40N + 4E bytes
```

### Pthreads
```
Memory = N × (sizeof(Node)) + E × sizeof(int) + thread overhead
Node structure: ~48 bytes (includes From_id array)
Total: ~48N + 8E bytes (stores both incoming and outgoing links)
```

### MPI (CSR)
```
Memory per process = (N/P) × sizeof(Node) + (E/P) × sizeof(int) + CSR overhead
CSR: row_ptr (4(N+1) bytes) + col_ind (4E bytes)
Total per process: ~40(N/P) + 4E/P + 4(N+1) + 4E bytes
```

**Example for 1M nodes, 10M edges, 4 processes:**
- Serial: ~80 MB
- Pthreads: ~96 MB
- MPI (per process): ~25 MB + shared CSR (~40 MB) = ~65 MB total

---

## When to Use Which Method?

### Use Serial When:
- ✅ Graph has < 100K nodes
- ✅ You're on a single-core machine
- ✅ You need a simple, debuggable implementation
- ✅ You're developing/testing

### Use Pthreads When:
- ✅ Graph has 100K - 10M nodes
- ✅ You have a multi-core machine (4-16 cores)
- ✅ You want good speedup without complexity
- ✅ All data fits in single machine memory

### Use MPI When:
- ✅ Graph has > 10M nodes
- ✅ You have access to a cluster
- ✅ Graph doesn't fit in single machine memory
- ✅ You need maximum scalability
- ✅ You're running production workloads

---

## Convergence Detection

All methods now use **adaptive convergence** with L1 norm:

**Serial:**
```c
l1_norm = sum of |p_t1[i] - p_t0[i]| for all i
if (l1_norm < threshold) break;
```

**Pthreads:**
```c
Each thread computes local_l1_norm
Threads accumulate: l1_norm += local_l1_norm (with mutex)
if (l1_norm < threshold) break;
```

**MPI:**
```c
Each process computes local_l1_norm
MPI_Allreduce: global_l1_norm = sum of all local_l1_norm
if (global_l1_norm < threshold) break;
```

**Why L1 Norm?**
- More robust than max error alone
- Accounts for overall change across all nodes
- Better convergence detection for large graphs

---

## Code Structure Differences

### Serial
```c
for (i = 0; i < N; i++) {
    // Process node i
    for (j = 0; j < Nodes[i].con_size; j++) {
        // Update neighbor
    }
}
```

### Pthreads
```c
void* Pagerank_Parallel(void* arg) {
    Thread *data = (Thread*)arg;
    for (i = data->start; i < data->end; i++) {
        // Process node i (thread's assigned range)
    }
}
// Main: Create threads, join threads
```

### MPI
```c
// Each process runs this:
for (i = local_start; i < local_end; i++) {
    // Process node i (process's assigned partition)
}
// Communicate with other processes
MPI_Allgatherv(...);
MPI_Allreduce(...);
```

---

## Summary Table

| Feature | Serial | Pthreads | MPI |
|---------|--------|----------|-----|
| **Parallelization** | None | Threads | Processes |
| **Memory Model** | Single | Shared | Distributed |
| **Communication** | None | Mutex locks | MPI messages |
| **Data Structure** | Adjacency lists | Adjacency lists | CSR format |
| **Scalability** | 1 core | CPU cores | Cluster size |
| **Complexity** | Low | Medium | High |
| **Best Graph Size** | < 100K | 100K-10M | > 10M |
| **Setup Required** | None | pthreads library | MPI runtime |

---

## Example: Same Algorithm, Different Execution

All three methods implement the same PageRank algorithm:

```
PR(v) = (1-d)/N + d × Σ(PR(u)/L(u))
```

Where:
- PR(v) = PageRank of node v
- d = damping factor (typically 0.85)
- N = total number of nodes
- L(u) = number of outgoing links from node u
- Σ = sum over all nodes u that link to v

The difference is **how** they compute this:
- **Serial**: Computes sequentially, one node at a time
- **Pthreads**: Computes in parallel, multiple nodes simultaneously on same machine
- **MPI**: Computes in parallel, multiple nodes simultaneously across multiple machines

