@echo off
REM Script to run the PageRank Dashboard

echo ========================================
echo PageRank Interactive Dashboard
echo ========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python is not installed or not in PATH!
    echo.
    echo Please install Python 3.8+ from: https://www.python.org/downloads/
    echo Make sure to check "Add Python to PATH" during installation.
    exit /b 1
)

echo Python found!
echo.

REM Check if we're in the right directory
if not exist "dashboard\app.py" (
    echo ERROR: dashboard\app.py not found!
    echo Make sure you're running this from the project root directory.
    exit /b 1
)

REM Check if requirements are installed
echo Checking if required packages are installed...
python -c "import streamlit, pandas, matplotlib, networkx, numpy" >nul 2>&1
if %errorlevel% neq 0 (
    echo.
    echo Required packages not found. Installing...
    echo.
    python -m pip install -r dashboard\requirements.txt
    if %errorlevel% neq 0 (
        echo.
        echo ERROR: Failed to install requirements!
        echo Please run manually: python -m pip install -r dashboard\requirements.txt
        exit /b 1
    )
    echo.
    echo Packages installed successfully!
    echo.
) else (
    echo Required packages are installed.
    echo.
)

REM Run the dashboard
echo Starting dashboard...
echo.
echo The dashboard will open in your web browser at: http://localhost:8501
echo.
echo Press Ctrl+C to stop the dashboard.
echo.
echo ========================================
echo.

cd dashboard
streamlit run app.py

cd ..

