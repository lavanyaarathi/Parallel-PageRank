# Dashboard Quick Start Guide

## Running the Dashboard on Windows

### Easiest Way (One Command)

```powershell
.\run_dashboard.bat
```

That's it! The script will:
1. ✅ Check if Python is installed
2. ✅ Install required packages automatically
3. ✅ Start the dashboard
4. ✅ Open it in your browser

### What You'll See

After running `.\run_dashboard.bat`, you'll see:

```
========================================
PageRank Interactive Dashboard
========================================

Python found!
Checking if required packages are installed...
Starting dashboard...

The dashboard will open in your web browser at: http://localhost:8501
```

Then your browser will automatically open to the dashboard!

## Using the Dashboard

### Step 1: Upload or Use Sample Graph
- Click "Upload File" to upload your own graph
- Or select "Use Sample Graph" for a quick test

### Step 2: Set Parameters
- **Number of Nodes**: How many nodes in your graph
- **Convergence Threshold**: When to stop (e.g., 0.0001)
- **Damping Factor (d)**: Usually 0.85

### Step 3: Choose Implementation
- **Serial**: Single-threaded (baseline)
- **Pthreads**: Multi-threaded (requires pthreads build)
- **MPI**: Distributed (requires MPI)

### Step 4: Set Threads/Processes
- For Pthreads: Number of threads (e.g., 4)
- For MPI: Number of processes (e.g., 4)

### Step 5: Run!
Click the **"🚀 Run PageRank"** button

### Step 6: View Results
The dashboard shows:
- ✅ Execution time and iterations
- ✅ Convergence plots (max error & L1 norm)
- ✅ Top-ranked nodes table
- ✅ PageRank distribution charts
- ✅ Graph structure visualization

## Troubleshooting

### "Python is not recognized"
**Fix**: Install Python from https://www.python.org/downloads/
- Make sure to check "Add Python to PATH" during installation
- Restart PowerShell after installation

### "ModuleNotFoundError: No module named 'streamlit'"
**Fix**: The script should install it automatically, but if not:
```powershell
pip install streamlit pandas matplotlib networkx numpy
```

### Dashboard won't start
**Fix**: 
1. Make sure port 8501 is not in use
2. Check if firewall is blocking it
3. Try manually: `cd dashboard && streamlit run app.py`

### "Executable not found" when running
**Fix**: Build the executables first:
```powershell
.\build_all.bat
```

## Stopping the Dashboard

Press `Ctrl+C` in the PowerShell window to stop the dashboard.

## Features

The dashboard provides:
- 📊 **Interactive Interface**: No command line needed!
- 📈 **Real-time Visualization**: See convergence as it happens
- 📉 **Multiple Charts**: Convergence, distribution, top nodes
- 🎨 **Graph Visualization**: Visual representation of your graph
- ⚡ **Quick Testing**: Easy parameter adjustment
- 💾 **Export Results**: View full output

## Example Workflow

1. **Start Dashboard**:
   ```powershell
   .\run_dashboard.bat
   ```

2. **In the Dashboard**:
   - Select "Use Sample Graph"
   - Set nodes: 4
   - Set threshold: 0.0001
   - Set d: 0.85
   - Choose "Serial"
   - Click "Run PageRank"

3. **View Results**:
   - See execution time
   - Check convergence plot
   - View top-ranked nodes
   - Explore graph visualization

4. **Try Different Methods**:
   - Run with Pthreads (if built)
   - Run with MPI
   - Compare results!

## Next Steps

- Try different graph files
- Experiment with parameters
- Compare Serial vs Pthreads vs MPI
- Use larger graphs for performance testing

Enjoy exploring PageRank! 🚀

