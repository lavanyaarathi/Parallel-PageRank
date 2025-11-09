# PageRank Dashboard

Interactive web dashboard for running and analyzing PageRank implementations.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
streamlit run app.py
```

The dashboard will open in your web browser at `http://localhost:8501`.

## Features

- Upload graph files or use sample graphs
- Run Serial, Pthreads, or MPI implementations
- Visualize convergence over iterations
- View top-ranked nodes
- Analyze PageRank distribution
- Graph structure visualization

## Requirements

- Python 3.8+
- Compiled PageRank executables in their respective directories
- For MPI: MPI runtime (mpiexec) must be available

