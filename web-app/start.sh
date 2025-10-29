#!/bin/bash
# Start script for Lung Cancer Prediction Web Application (Linux/Mac)
# Usage: chmod +x start.sh && ./start.sh

echo "================================================================================"
echo "LUNG CANCER PREDICTION WEB APPLICATION - STARTUP SCRIPT"
echo "================================================================================"
echo ""

# Use existing virtual environment from parent directory
if [ ! -d "../venv" ]; then
    echo "ERROR: Virtual environment not found at ../venv"
    echo "Please create it first by running:"
    echo "  cd .."
    echo "  python3 -m venv venv"
    echo "  source venv/bin/activate"
    echo "  pip install -r requirements.txt"
    exit 1
fi

# Activate virtual environment
echo "Activating virtual environment..."
source ../venv/bin/activate

# Check if Flask is installed
if ! python -c "import flask" 2>/dev/null; then
    echo "Flask not found. Installing web dependencies..."
    pip install -r ../requirements.txt
    echo ""
fi

# Start the application
echo "Starting web application..."
echo ""
python backend/app.py

# Deactivate on exit
deactivate

