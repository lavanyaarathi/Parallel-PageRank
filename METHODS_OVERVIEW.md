# PageRank Methods Overview

A visual guide to understanding the different PageRank calculation methods.

## 🎯 Three Ways to Calculate PageRank

All three methods implement the **same PageRank algorithm**, but use different parallelization strategies:

```
PR(v) = (1-d)/N + d × Σ(PR(u)/L(u))
```

The difference is **HOW** they compute it.

---

## 📊 Method Comparison at a Glance

```
┌─────────────────────────────────────────────────────────────────┐
│                    PAGERANK IMPLEMENTATIONS                     │
└─────────────────────────────────────────────────────────────────┘

┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│   SERIAL     │  │  PTHREADS    │  │     MPI      │
├──────────────┤  ├──────────────┤  ├──────────────┤
│              │  │              │  │              │
│  [Node 1]    │  │ Thread 1:    │  │ Process 1:   │
│  [Node 2]    │  │ [Node 1-2]   │  │ [Node 1-2]   │
│  [Node 3]    │  │              │  │              │
│  [Node 4]    │  │ Thread 2:    │  │ Process 2:   │
│              │  │ [Node 3-4]   │  │ [Node 3-4]   │
│  Sequential  │  │              │  │              │
│  Processing  │  │ Parallel     │  │ Distributed  │
│              │  │ (Shared Mem) │  │ (Network)    │
│              │  │              │  │              │
│  1 Core      │  │ 4-8 Cores    │  │ Many Nodes   │
│  1x Speed    │  │ 2-8x Speed   │  │ 4-100x Speed │
└──────────────┘  └──────────────┘  └──────────────┘
```

---

## 🔄 Execution Flow Comparison

### Serial Method
```
Start
  ↓
Read Graph
  ↓
Initialize PR values
  ↓
┌─────────────────┐
│ For each node:  │ ← Sequential loop
│  Compute PR     │
│  Update values  │
└─────────────────┘
  ↓
Check convergence
  ↓
[Not converged?] → Yes → Loop back
  ↓ No
Output results
  ↓
End
```

### Pthreads Method
```
Start
  ↓
Read Graph
  ↓
Initialize PR values
  ↓
Create N threads
  ↓
┌─────────────────────────────────┐
│ Thread 1: Process nodes 1-2    │
│ Thread 2: Process nodes 3-4    │ ← Parallel execution
│ Thread 3: Process nodes 5-6    │
│ Thread 4: Process nodes 7-8    │
└─────────────────────────────────┘
  ↓
Synchronize (mutex locks)
  ↓
Check convergence
  ↓
[Not converged?] → Yes → Loop back
  ↓ No
Output results
  ↓
End
```

### MPI Method
```
Start (on each process)
  ↓
Process 0: Read Graph → Convert to CSR
  ↓
Broadcast graph to all processes
  ↓
Each process gets partition
  ↓
┌─────────────────────────────────┐
│ Process 1: Process nodes 1-2   │
│ Process 2: Process nodes 3-4   │ ← Distributed execution
│ Process 3: Process nodes 5-6   │
│ Process 4: Process nodes 7-8   │
└─────────────────────────────────┘
  ↓
MPI_Allgatherv (exchange values)
  ↓
MPI_Allreduce (compute global sums)
  ↓
Check convergence
  ↓
[Not converged?] → Yes → Loop back
  ↓ No
Output results
  ↓
End
```

---

## 💾 Memory Model Comparison

### Serial
```
┌─────────────────────────────┐
│   Single Memory Space       │
│                             │
│  [All Nodes]                │
│  [All Edges]                │
│  [All PR values]            │
│                             │
│  One process accesses all   │
└─────────────────────────────┘
```

### Pthreads
```
┌─────────────────────────────┐
│   Shared Memory Space       │
│                             │
│  [All Nodes] ───────────┐   │
│  [All Edges]            │   │
│  [All PR values]        │   │
│                         │   │
│  Thread 1 ──────────────┘   │
│  Thread 2 ──────────────┘   │
│  Thread 3 ──────────────┘   │
│  Thread 4 ──────────────┘   │
│                             │
│  All threads share memory   │
│  (use mutex for safety)     │
└─────────────────────────────┘
```

### MPI
```
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│  Process 1   │  │  Process 2   │  │  Process 3   │
│              │  │              │  │              │
│ [Nodes 1-2]  │  │ [Nodes 3-4]  │  │ [Nodes 5-6]  │
│ [Edges 1-2]  │  │ [Edges 3-4]  │  │ [Edges 5-6]  │
│ [PR 1-2]     │  │ [PR 3-4]     │  │ [PR 5-6]     │
│              │  │              │  │              │
└──────┬───────┘  └──────┬───────┘  └──────┬───────┘
       │                 │                 │
       └─────────────────┼─────────────────┘
                         │
              Network Communication
              (MPI messages)
```

---

## ⚡ Performance Characteristics

### Speedup Comparison

```
Speedup
  ↑
  │                                    MPI (ideal)
  │                                  ╱
  │                                ╱
  │                              ╱
  │                            ╱
  │                          ╱
  │                        ╱
  │                      ╱
  │                    ╱
  │                  ╱
  │                ╱
  │              ╱
  │            ╱
  │          ╱
  │        ╱
  │      ╱  Pthreads (typical)
  │    ╱
  │  ╱
  │╱
  └────────────────────────────────────→ Processors
  1    2    4    8   16   32   64
```

### When to Use Each Method

```
Graph Size          Recommended Method
─────────────────────────────────────────
< 100K nodes    →   Serial
100K - 10M      →   Pthreads
> 10M nodes     →   MPI
```

---

## 🔧 Technical Differences

### Data Structures

| Method | Graph Storage | PR Storage | Communication |
|--------|--------------|------------|---------------|
| Serial | Adjacency lists | Array | None |
| Pthreads | Adjacency lists (bidirectional) | Array (shared) | Mutex locks |
| MPI | CSR format | Distributed arrays | MPI messages |

### Synchronization

| Method | Sync Mechanism | Overhead |
|--------|---------------|----------|
| Serial | None | None |
| Pthreads | Mutex locks, barriers | Low |
| MPI | MPI_Allgatherv, MPI_Allreduce | Medium-High |

### Scalability

| Method | Max Processors | Bottleneck |
|--------|---------------|------------|
| Serial | 1 | CPU |
| Pthreads | CPU cores (4-64) | Memory bandwidth |
| MPI | Cluster size (100s-1000s) | Network latency |

---

## 📈 Example: Same Problem, Different Approaches

**Problem**: Calculate PageRank for 1,000,000 nodes

### Serial Approach
```
Time: 100 seconds
Memory: 80 MB
Processors: 1
```

### Pthreads Approach
```
Time: 15 seconds (6.7x speedup)
Memory: 96 MB
Processors: 8 cores
```

### MPI Approach
```
Time: 2 seconds (50x speedup)
Memory: 20 MB per process (160 MB total for 8 processes)
Processors: 8 nodes (64 cores total)
```

---

## 🎓 Key Takeaways

1. **Same Algorithm**: All three implement identical PageRank formula
2. **Different Execution**: How they parallelize differs
3. **Trade-offs**: 
   - Serial: Simple but slow
   - Pthreads: Good speedup, single machine
   - MPI: Best speedup, requires cluster
4. **Choose Based On**:
   - Graph size
   - Available hardware
   - Performance requirements

---

## 🚀 Quick Start

**See the differences in action:**

```bash
# Compare all three methods
python compare_methods.py pagerank_mpi/small_graph.txt 4 0.0001 0.85 4
```

**Learn more:**
- [QUICKSTART.md](QUICKSTART.md) - How to run the project
- [COMPARISON.md](COMPARISON.md) - Detailed technical comparison
- [FEATURES.md](FEATURES.md) - All features and optimizations

