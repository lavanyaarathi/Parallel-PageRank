@echo off
echo Testing Simulated Distributed PageRank...

REM Check if executable exists
if not exist pagerank_simulated.exe (
    echo Error: pagerank_simulated.exe not found. Please build first with build.bat
    exit /b 1
)

REM Create test graph if it doesn't exist
if not exist small_graph.txt (
    echo Creating small test graph...
    echo # Small test graph > small_graph.txt
    echo 0	1 >> small_graph.txt
    echo 0	2 >> small_graph.txt
    echo 1	2 >> small_graph.txt
    echo 2	3 >> small_graph.txt
    echo 3	0 >> small_graph.txt
    echo 3	1 >> small_graph.txt
)

echo.
echo Running tests...
echo.

REM Test 1: Basic test with 2 processes
echo Test 1: Running with 2 processes...
pagerank_simulated.exe 2 4 0.001
if %errorlevel% neq 0 (
    echo Test 1 failed!
    exit /b 1
)
echo Test 1 passed!
echo.

REM Test 2: Test with 4 processes
echo Test 2: Running with 4 processes...
pagerank_simulated.exe 4 4 0.001
if %errorlevel% neq 0 (
    echo Test 2 failed!
    exit /b 1
)
echo Test 2 passed!
echo.

REM Test 3: Test with custom parameters
echo Test 3: Running with custom parameters...
pagerank_simulated.exe 3 10 0.0001 0.9
if %errorlevel% neq 0 (
    echo Test 3 failed!
    exit /b 1
)
echo Test 3 passed!
echo.

echo All tests passed successfully!