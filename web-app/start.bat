@echo off
REM Start script for Lung Cancer Prediction Web Application (Windows)
REM Usage: double-click this file or run: .\start.bat

echo ================================================================================
echo LUNG CANCER PREDICTION WEB APPLICATION - STARTUP SCRIPT
echo ================================================================================
echo.

REM Use existing virtual environment from parent directory
if not exist "..\venv\" (
    echo ERROR: Virtual environment not found at ..\venv
    echo Please create it first by running:
    echo   cd ..
    echo   python -m venv venv
    echo   venv\Scripts\activate
    echo   pip install -r requirements.txt
    pause
    exit /b 1
)

REM Activate virtual environment
echo Activating virtual environment...
call ..\venv\Scripts\activate.bat

REM Check if Flask is installed
pip show flask >nul 2>&1
if errorlevel 1 (
    echo Flask not found. Installing web dependencies...
    pip install -r ..\requirements.txt
    echo.
)

REM Start the application
echo Starting web application...
echo.
python backend\app.py

REM Keep window open if there's an error
if errorlevel 1 (
    echo.
    echo ================================================================================
    echo ERROR: Application failed to start
    echo ================================================================================
    pause
)

