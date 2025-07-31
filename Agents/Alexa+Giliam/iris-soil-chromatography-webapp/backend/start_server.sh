#!/bin/bash

# Chromatography Analysis Backend Startup Script

echo "🔬 Starting Chromatography Analysis Backend..."
echo "============================================="

# Check if we're in the right directory
if [ ! -f "api.py" ]; then
    echo "❌ Error: api.py not found. Please run this script from the backend directory."
    exit 1
fi

# Activate conda environment if available
if command -v conda &> /dev/null; then
    echo "🐍 Activating conda environment: summerschoolenv"
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate summerschoolenv
    if [ $? -ne 0 ]; then
        echo "⚠️  Failed to activate summerschoolenv. Creating environment..."
        conda create -n summerschoolenv python=3.9 -y
        conda activate summerschoolenv
    fi
    PYTHON_CMD="python"
else
    # Check if Python is available
    if ! command -v python3 &> /dev/null && ! command -v python &> /dev/null; then
        echo "❌ Error: Python not found. Please install Python 3.7 or higher."
        exit 1
    fi

    # Use python3 if available, otherwise python
    PYTHON_CMD="python3"
    if ! command -v python3 &> /dev/null; then
        PYTHON_CMD="python"
    fi
fi

echo "🐍 Using Python: $PYTHON_CMD"

# Check if requirements are installed
echo "📦 Checking dependencies..."
$PYTHON_CMD -c "import flask, cv2, numpy, sklearn, scipy" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "⚠️  Some dependencies are missing. Installing requirements..."
    $PYTHON_CMD -m pip install -r requirements.txt
    if [ $? -ne 0 ]; then
        echo "❌ Failed to install dependencies. Please check your Python/pip installation."
        exit 1
    fi
fi

# Create uploads directory if it doesn't exist
if [ ! -d "uploads" ]; then
    echo "📁 Creating uploads directory..."
    mkdir uploads
fi

echo "✅ All checks passed!"
echo "🚀 Starting API server on http://localhost:8080"
echo "💡 Press Ctrl+C to stop the server"
echo ""

# Start the API server
$PYTHON_CMD api.py
