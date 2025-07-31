#!/bin/bash

# Conda Environment Setup Script for Chromatography Analysis

echo "🐍 Setting up Conda Environment for Chromatography Analysis"
echo "=========================================================="

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "❌ Error: Conda not found. Please install Miniconda or Anaconda first."
    echo "   Download from: https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

# Initialize conda for this shell session
source "$(conda info --base)/etc/profile.d/conda.sh"

# Check if summerschoolenv already exists
if conda env list | grep -q "summerschoolenv"; then
    echo "✅ Environment 'summerschoolenv' already exists."
    echo "🔄 Activating environment..."
    conda activate summerschoolenv
else
    echo "🆕 Creating new conda environment: summerschoolenv"
    echo "   Python version: 3.9"
    conda create -n summerschoolenv python=3.9 -y
    
    if [ $? -ne 0 ]; then
        echo "❌ Failed to create conda environment."
        exit 1
    fi
    
    echo "🔄 Activating environment..."
    conda activate summerschoolenv
fi

# Check if we're in the backend directory, if not navigate there
if [ ! -f "requirements.txt" ]; then
    if [ -f "backend/requirements.txt" ]; then
        cd backend
    else
        echo "❌ Error: requirements.txt not found. Please run this script from the project root or backend directory."
        exit 1
    fi
fi

echo "📦 Installing Python dependencies..."
echo "   Installing from requirements.txt"

# Install pip packages
pip install -r requirements.txt

if [ $? -ne 0 ]; then
    echo "❌ Failed to install some packages. Trying with conda-forge..."
    
    # Try installing main packages with conda first
    echo "   Installing core packages with conda..."
    conda install -c conda-forge flask numpy opencv scikit-image scikit-learn scipy matplotlib pandas -y
    
    # Then install remaining with pip
    echo "   Installing remaining packages with pip..."
    pip install flask-cors requests pywavelets
fi

echo ""
echo "✅ Environment setup complete!"
echo "📋 Environment Summary:"
echo "   Name: summerschoolenv"
echo "   Python: $(python --version)"
echo "   Location: $(which python)"

echo ""
echo "🚀 To use this environment:"
echo "   conda activate summerschoolenv"
echo ""
echo "🔧 To start the backend server:"
echo "   ./start_server.sh"
echo ""
echo "📚 Installed packages:"
pip list | grep -E "(flask|numpy|opencv|scikit|scipy|matplotlib|pandas|requests)"

echo ""
echo "💡 Note: This environment is now active in this terminal session."
