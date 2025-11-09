# Windows Setup Guide

This guide will help you set up and run the PageRank project on Windows.

## Prerequisites

### 1. Install GCC Compiler

You need a C compiler. Choose one:

**Option A: MinGW-w64 (Recommended)**
1. Download from: https://www.mingw-w64.org/downloads/
2. Or use installer: https://sourceforge.net/projects/mingw-w64/
3. Add `C:\mingw64\bin` to your PATH

**Option B: MSYS2 (Easier installation)**
1. Download from: https://www.msys2.org/
2. Install and run: `pacman -S mingw-w64-x86_64-gcc`
3. Add `C:\msys64\mingw64\bin` to your PATH

**Option C: Visual Studio (Alternative)**
1. Install Visual Studio with C/C++ support
2. Use Developer Command Prompt

### 2. Install Microsoft MPI (Already Installed ✓)

You mentioned you already have MPI installed. If not:
1. Download from: https://www.microsoft.com/en-us/download/details.aspx?id=57467
2. Install both:
   - `msmpisdk.msi` (SDK)
   - `msmpisetup.exe` (Runtime)

### 3. Install Python 3.8+ (Optional, for tools)

1. Download from: https://www.python.org/downloads/
2. Make sure to check "Add Python to PATH" during installation

## Building the Project

### Quick Build (All at Once)

```powershell
.\build_all.bat
```

This will build all three implementations (Serial, Pthreads, MPI).

### Individual Builds

**Serial:**
```powershell
cd pagerank_serial
.\build.bat
cd ..
```

**Pthreads:**
```powershell
cd pagerank_pthreads
.\build.bat
cd ..
```

**MPI:**
```powershell
cd pagerank_mpi
.\build.bat
cd ..
```

## Running the Programs

### Serial Implementation

```powershell
.\pagerank_serial\pagerank_serial.exe pagerank_mpi\small_graph.txt 4 0.0001 0.85
```

### Pthreads Implementation

```powershell
.\pagerank_pthreads\pagerank_pthreads.exe pagerank_mpi\small_graph.txt 4 0.0001 0.85 4
```

### MPI Implementation

```powershell
mpiexec -n 4 .\pagerank_mpi\pagerank_mpi.exe pagerank_mpi\small_graph.txt 4 0.0001 0.85
```

**Note**: On Windows, use `mpiexec` (not `mpirun`)

## Compare All Methods

```powershell
python compare_methods.py pagerank_mpi\small_graph.txt 4 0.0001 0.85 4
```

## Troubleshooting

### Issue: "gcc is not recognized"

**Solution**: 
1. Install MinGW-w64 or MSYS2 (see Prerequisites)
2. Add GCC to PATH:
   - Right-click "This PC" → Properties → Advanced System Settings
   - Environment Variables → System Variables → Path → Edit
   - Add: `C:\mingw64\bin` (or your GCC path)
3. Restart PowerShell

**Verify installation:**
```powershell
gcc --version
```

### Issue: "mpiexec is not recognized"

**Solution**:
1. Add MPI to PATH:
   - Add: `C:\Program Files\Microsoft MPI\Bin`
   - Or: `C:\Program Files (x86)\Microsoft SDKs\MPI\Bin\x64`
2. Restart PowerShell

**Verify installation:**
```powershell
mpiexec
```

### Issue: "pthreads not found" or linking errors

**Solution**:
1. Make sure you're using MinGW-w64 (not MSVC)
2. MinGW-w64 includes pthreads support
3. If using MSYS2, install: `pacman -S mingw-w64-x86_64-pthreads`

### Issue: MPI build fails with "mpi.h not found"

**Solution**:
1. Check if MPI is installed:
   ```powershell
   Test-Path "C:\Program Files (x86)\Microsoft SDKs\MPI\Include\mpi.h"
   ```
2. If not found, install Microsoft MPI (see Prerequisites)
3. The build script should auto-detect, but if it doesn't, edit `pagerank_mpi\build.bat` and update the paths

### Issue: MPI build fails with "sal.h not found" or path errors

**Solution**:
This means Windows SDK is missing. Microsoft MPI requires Windows SDK headers.

**Quick Fix**:
1. Install Windows 10 SDK: https://developer.microsoft.com/en-us/windows/downloads/windows-sdk/
2. Or install Visual Studio Community (includes SDK): https://visualstudio.microsoft.com/downloads/
3. Rebuild using the working script: `cd pagerank_mpi && .\build_working.bat`

**If you get path errors** (like "\Microsoft was unexpected"):
- Use `build_working.bat` instead of `build.bat` or `build_simple.bat`
- This script avoids path variable expansion issues

**Alternative**: Use Developer Command Prompt (if you have Visual Studio):
- Open "Developer Command Prompt for VS"
- Navigate to project and build

**See [WINDOWS_MPI_FIX.md](WINDOWS_MPI_FIX.md) for detailed instructions.**

### Issue: "The system cannot find the path specified"

**Solution**:
- Make sure you're in the project root directory
- Use full paths if needed:
  ```powershell
  cd "C:\Users\Lavanya Rathi\Downloads\parallel-pagerank-master\parallel-pagerank-master"
  ```

### Issue: Permission denied

**Solution**:
- Run PowerShell as Administrator
- Or check file permissions

## Using Visual Studio (Alternative)

If you prefer Visual Studio:

1. Open Visual Studio
2. Create new C++ project
3. Add source files
4. Configure:
   - **Serial**: No special libraries needed
   - **Pthreads**: Link against pthread library (may need pthreads-win32)
   - **MPI**: Add MPI include and library paths

## Quick Test

After building, test with:

```powershell
# Test Serial
.\pagerank_serial\pagerank_serial.exe pagerank_mpi\small_graph.txt 4 0.0001 0.85

# Should output:
# Iteration 1: Max Error = ..., L1 Norm = ...
# ...
# Total iterations: X
# Totaltime = X seconds
```

## Running the Dashboard (Frontend)

The interactive web dashboard lets you run PageRank through a browser interface.

### Quick Start

```powershell
.\run_dashboard.bat
```

This will:
- Check Python installation
- Install required packages
- Start the dashboard
- Open it in your browser at http://localhost:8501

### Manual Setup

1. **Install Python packages**:
   ```powershell
   cd dashboard
   pip install -r requirements.txt
   cd ..
   ```

2. **Run dashboard**:
   ```powershell
   cd dashboard
   streamlit run app.py
   ```

3. **Open browser**: http://localhost:8501

**For detailed dashboard instructions**: See [dashboard/README_WINDOWS.md](dashboard/README_WINDOWS.md)

## Next Steps

1. ✅ Build all implementations: `.\build_all.bat`
2. ✅ Test each one individually
3. ✅ Run comparison: `python compare_methods.py pagerank_mpi\small_graph.txt 4 0.0001 0.85 4`
4. ✅ Run dashboard: `.\run_dashboard.bat`
5. 📖 Read [QUICKSTART.md](QUICKSTART.md) for more details
6. 📖 Read [COMPARISON.md](COMPARISON.md) to understand differences

## Getting Help

If you encounter issues:
1. Check this troubleshooting section
2. Verify all prerequisites are installed
3. Make sure PATH variables are set correctly
4. Try building individually to isolate the problem

