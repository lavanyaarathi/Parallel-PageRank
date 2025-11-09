@echo off
setlocal enabledelayedexpansion

echo Building MPI PageRank implementation...

REM Clean previous build
if exist pagerank_mpi.exe del pagerank_mpi.exe
if exist pagerank_mpi.o del pagerank_mpi.o
if exist csr_graph.o del csr_graph.o

REM Check MPI installation
set "MPI_INC=C:\Program Files (x86)\Microsoft SDKs\MPI\Include"
set "MPI_LIB=C:\Program Files (x86)\Microsoft SDKs\MPI\Lib\x64"

if not exist "%MPI_INC%\mpi.h" (
    echo ERROR: MPI installation not found!
    exit /b 1
)

echo Found Microsoft MPI installation

REM Check Windows SDK
set "SDK_VER=10.0.19041.0"
set "SDK_INC=C:\Program Files (x86)\Windows Kits\10\Include\%SDK_VER%\shared"

if not exist "%SDK_INC%\sal.h" (
    set "SDK_VER=10.0.22621.0"
    set "SDK_INC=C:\Program Files (x86)\Windows Kits\10\Include\%SDK_VER%\shared"
    if not exist "%SDK_INC%\sal.h" (
        set "SDK_VER=10.0.18362.0"
        set "SDK_INC=C:\Program Files (x86)\Windows Kits\10\Include\%SDK_VER%\shared"
        if not exist "%SDK_INC%\sal.h" (
            echo ERROR: Windows SDK not found!
            exit /b 1
        )
    )
)

echo Found Windows SDK version %SDK_VER%

REM Use MSYS2 GCC if available
set "GCC_CMD=C:\msys64\mingw64\bin\gcc.exe"
if not exist "!GCC_CMD!" (
    set "GCC_CMD=gcc"
)

echo Using GCC: !GCC_CMD!
echo.

REM Compile pagerank_mpi.c
echo Compiling pagerank_mpi.c...
"!GCC_CMD!" -I"%MPI_INC%" -I"%SDK_INC%" -D__USE_MINGW_ANSI_STDIO=1 -Wall -O3 -std=c99 -c pagerank_mpi.c -o pagerank_mpi.o
if errorlevel 1 (
    echo ERROR: Compilation of pagerank_mpi.c failed!
    exit /b 1
)

REM Compile csr_graph.c
echo Compiling csr_graph.c...
"!GCC_CMD!" -I"%MPI_INC%" -I"%SDK_INC%" -D__USE_MINGW_ANSI_STDIO=1 -Wall -O3 -std=c99 -c csr_graph.c -o csr_graph.o
if errorlevel 1 (
    echo ERROR: Compilation of csr_graph.c failed!
    exit /b 1
)

REM Link
echo Linking...
"!GCC_CMD!" pagerank_mpi.o csr_graph.o -L"%MPI_LIB%" -lmsmpi -lm -o pagerank_mpi.exe
if errorlevel 1 (
    echo ERROR: Linking failed!
    exit /b 1
)

echo.
echo Build successful!
echo.
echo To run: mpiexec -n 4 pagerank_mpi.exe small_graph.txt 4 0.0001 0.85
echo.

