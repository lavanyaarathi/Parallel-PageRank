@echo off
REM Build script for MPI PageRank on Windows

echo Building MPI PageRank implementation...

REM Check if mpicc is available
where mpicc >nul 2>nul
if %errorlevel% neq 0 (
    echo Error: mpicc not found. Please ensure MPI is installed and in PATH.
    echo You may need to install Microsoft MPI or Intel MPI.
    exit /b 1
)

REM Clean previous build
if exist pagerank_mpi.exe del pagerank_mpi.exe
if exist pagerank_mpi.o del pagerank_mpi.o

REM Compile
echo Compiling pagerank_mpi.c...
gcc -I"C:\Program Files (x86)\Microsoft SDKs\MPI\Include" -Wall -O3 -std=c99 -c pagerank_mpi.c -o pagerank_mpi.o

if %errorlevel% neq 0 (
    echo Compilation failed!
    exit /b 1
)

REM Link
echo Linking...
gcc pagerank_mpi.o -L"C:\Program Files (x86)\Microsoft SDKs\MPI\Lib\x64" -lmsmpi -lm -o pagerank_mpi.exe
if %errorlevel% neq 0 (
    Linking failed!
    exit /b 1
)

echo Build successful!
echo.
echo To run the program:
echo   mpirun -np 4 pagerank_mpi.exe web-Google.txt 1000 0.0001 0.85
echo.
echo Note: You need to have a graph file (e.g., web-Google.txt) in the current directory