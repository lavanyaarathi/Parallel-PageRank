@echo off
REM Test script for MPI PageRank on Windows

echo MPI PageRank Test Script
echo ========================
echo.

REM Check if mpirun is available
where mpirun >nul 2>nul
if %errorlevel% neq 0 (
    echo Error: mpirun not found. Please ensure Microsoft MPI or Intel MPI is installed.
    exit /b 1
)

REM Check if executable exists
if not exist "pagerank_mpi.exe" (
    echo Error: pagerank_mpi.exe not found. Please compile first using build.bat
    exit /b 1
)

REM Create a small test graph if web-Google.txt doesn't exist
if not exist "web-Google.txt" (
    echo Creating small test graph...
    (
        echo # Small test graph
        echo 0    1
        echo 0    2
        echo 1    2
        echo 2    0
        echo 3    0
        echo 3    1
        echo 3    2
    ) > small_graph.txt
    set GRAPH_FILE=small_graph.txt
    set NODES=4
) else (
    set GRAPH_FILE=web-Google.txt
    set NODES=1000
)

echo Using graph file: %GRAPH_FILE%
echo Number of nodes: %NODES%
echo.

REM Test with different process counts
for %%p in (2 4) do (
    echo Testing with %%p processes...
    echo -------------------------
    
    mpirun -np %%p pagerank_mpi.exe %GRAPH_FILE% %NODES% 0.001 0.85
    
    if %errorlevel% equ 0 (
        echo [OK] Test with %%p processes PASSED
    ) else (
        echo [FAIL] Test with %%p processes FAILED
    )
    echo.
)

echo Test completed!
echo.
echo For larger graphs, you can use:
echo   mpirun -np 8 pagerank_mpi.exe web-Google.txt 10000 0.0001 0.85
pause