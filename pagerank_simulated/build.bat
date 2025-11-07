@echo off
echo Building Simulated Distributed PageRank...

REM Check for Visual Studio compiler
where cl >nul 2>nul
if %errorlevel% neq 0 (
    echo Error: Visual Studio compiler not found.
    echo Please install Visual Studio or Visual Studio Build Tools
    echo and run this script from a Visual Studio Developer Command Prompt.
    exit /b 1
)

REM Clean previous build
if exist pagerank_simulated.exe del pagerank_simulated.exe
if exist *.obj del *.obj

REM Compile the program
echo Compiling pagerank_simulated.c...
cl /O2 /Fe:pagerank_simulated.exe pagerank_simulated.c

if %errorlevel% neq 0 (
    echo Build failed!
    exit /b 1
)

echo Build successful!
echo.
echo Usage examples:
echo   pagerank_simulated.exe              - Run with 2 processes (default)
echo   pagerank_simulated.exe 4            - Run with 4 processes
echo   pagerank_simulated.exe 4 10 0.001     - Run with 4 processes, 10 nodes, threshold 0.001
echo   pagerank_simulated.exe 4 10 0.001 0.9 - Run with 4 processes, 10 nodes, threshold 0.001, damping 0.9
echo.
echo To test with the small graph:
echo   pagerank_simulated.exe 2 4 0.001 small_graph.txt