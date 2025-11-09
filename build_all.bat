@echo off
REM Build all PageRank implementations on Windows

echo ========================================
echo Building All PageRank Implementations
echo ========================================
echo.

REM Build Serial
echo [1/3] Building Serial implementation...
cd pagerank_serial
call build.bat
if %errorlevel% neq 0 (
    echo Serial build failed!
    cd ..
    exit /b 1
)
cd ..
echo.

REM Build Pthreads
echo [2/3] Building Pthreads implementation...
cd pagerank_pthreads
call build.bat
if %errorlevel% neq 0 (
    echo.
    echo WARNING: Pthreads build failed!
    echo You can still use Serial and MPI implementations.
    echo See WINDOWS_SETUP.md for help installing pthreads support.
    echo.
    cd ..
) else (
    cd ..
)
echo.

REM Build MPI
echo [3/3] Building MPI implementation...
cd pagerank_mpi
call build.bat
if %errorlevel% neq 0 (
    echo.
    echo WARNING: MPI build failed!
    echo This is usually because Windows SDK is not installed.
    echo See WINDOWS_SETUP.md for solutions.
    echo.
    echo You can still use the Serial implementation.
    echo.
    cd ..
) else (
    cd ..
)
echo.

echo ========================================
echo All builds completed successfully!
echo ========================================
echo.
echo To test the implementations:
echo   python compare_methods.py pagerank_mpi\small_graph.txt 4 0.0001 0.85 4
echo.

