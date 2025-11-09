# Running the Dashboard on Windows

## Quick Start

### Option 1: Use the Batch Script (Easiest)

```powershell
.\run_dashboard.bat
```

This script will:
- Check if Python is installed
- Install required packages if needed
- Start the dashboard
- Open it in your browser

### Option 2: Manual Setup

1. **Install Python** (if not already installed):
   - Download from: https://www.python.org/downloads/
   - Make sure to check "Add Python to PATH" during installation

2. **Install required packages**:
   ```powershell
   cd dashboard
   pip install -r requirements.txt
   cd ..
   ```

3. **Run the dashboard**:
   ```powershell
   cd dashboard
   streamlit run app.py
   cd ..
   ```

4. **Open in browser**:
   - The dashboard will automatically open at: http://localhost:8501
   - If it doesn't, manually open: http://localhost:8501

## Using the Dashboard

1. **Upload a graph file** or use the sample graph
2. **Select parameters**:
   - Number of nodes
   - Convergence threshold
   - Damping factor (d)
3. **Choose implementation**: Serial, Pthreads, or MPI
4. **Set threads/processes** (for Pthreads/MPI)
5. **Click "Run PageRank"**
6. **View results**:
   - Execution time and iterations
   - Convergence plots
   - Top-ranked nodes
   - PageRank distribution
   - Graph visualization

## Troubleshooting

### Issue: "Python is not recognized"

**Solution**:
1. Install Python from https://www.python.org/downloads/
2. During installation, check "Add Python to PATH"
3. Restart PowerShell
4. Verify: `python --version`

### Issue: "pip is not recognized"

**Solution**:
1. Python 3.4+ includes pip
2. If missing, install pip: `python -m ensurepip --upgrade`
3. Or reinstall Python with pip option checked

### Issue: "streamlit is not recognized"

**Solution**:
```powershell
pip install streamlit
```

### Issue: "ModuleNotFoundError"

**Solution**:
```powershell
cd dashboard
pip install -r requirements.txt
```

### Issue: Dashboard won't start

**Solution**:
1. Make sure port 8501 is not in use
2. Try a different port: `streamlit run app.py --server.port 8502`
3. Check firewall settings

### Issue: Executables not found when running

**Solution**:
1. Make sure you've built the executables first:
   ```powershell
   .\build_all.bat
   ```
2. The dashboard looks for executables in:
   - `pagerank_serial\pagerank_serial.exe`
   - `pagerank_pthreads\pagerank_pthreads.exe`
   - `pagerank_mpi\pagerank_mpi.exe`

## Features

The dashboard provides:
- ✅ Graph file upload
- ✅ Parameter configuration
- ✅ Run any implementation (Serial/Pthreads/MPI)
- ✅ Real-time convergence visualization
- ✅ Top-ranked nodes display
- ✅ PageRank distribution charts
- ✅ Graph structure visualization
- ✅ Execution time and iteration metrics

## Stopping the Dashboard

Press `Ctrl+C` in the terminal to stop the dashboard server.

