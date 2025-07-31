#!/bin/bash

# Chromatography Analysis - Complete Startup Script

echo "🔬 Chromatography Analysis Tool"
echo "==============================="
echo ""

# Check if we're in the project root
if [ ! -d "frontend" ] || [ ! -d "backend" ]; then
    echo "❌ Error: Please run this script from the project root directory."
    echo "   Expected structure:"
    echo "     project-root/"
    echo "     ├── frontend/"
    echo "     └── backend/"
    exit 1
fi

# Function to check if a port is in use
port_in_use() {
    lsof -Pi :$1 -sTCP:LISTEN -t >/dev/null 2>&1
}

echo "🚀 Starting Chromatography Analysis Tool..."
echo ""

# Start backend
echo "1️⃣  Starting Backend API Server..."
echo "   📁 Directory: backend/"
echo "   🌐 URL: http://localhost:8080"

cd backend

# Activate conda environment if available
if command -v conda &> /dev/null; then
    echo "   🐍 Activating conda environment: summerschoolenv"
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate summerschoolenv
    if [ $? -ne 0 ]; then
        echo "   ⚠️  Failed to activate summerschoolenv. Creating environment..."
        conda create -n summerschoolenv python=3.9 -y
        conda activate summerschoolenv
    fi
    PYTHON_CMD="python"
else
    # Check if backend dependencies are installed
    if ! command -v python3 &> /dev/null && ! command -v python &> /dev/null; then
        echo "❌ Error: Python not found. Please install Python 3.7 or higher."
        exit 1
    fi

    PYTHON_CMD="python3"
    if ! command -v python3 &> /dev/null; then
        PYTHON_CMD="python"
    fi
fi

# Quick dependency check
$PYTHON_CMD -c "import flask" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "📦 Installing backend dependencies..."
    $PYTHON_CMD -m pip install -r requirements.txt
fi

# Start backend in background
echo "   🔧 Starting API server..."
$PYTHON_CMD api.py > ../backend.log 2>&1 &
BACKEND_PID=$!
echo "   ✅ Backend started (PID: $BACKEND_PID)"

# Wait a moment for backend to start
echo "   ⏳ Waiting for backend to initialize..."
sleep 5

# Check if backend is running
if ! port_in_use 8080; then
    echo "❌ Backend failed to start on port 8080. Checking backend.log..."
    echo "📄 Last 10 lines of backend.log:"
    tail -10 ../backend.log
    exit 1
else
    echo "   ✅ Backend is responding on port 8080"
fi

cd ..

echo ""
echo "2️⃣  Starting Frontend Web Server..."
echo "   📁 Directory: frontend/"

cd frontend

# Find available port for frontend
FRONTEND_PORT=3000
while port_in_use $FRONTEND_PORT; do
    FRONTEND_PORT=$((FRONTEND_PORT + 1))
    if [ $FRONTEND_PORT -gt 3010 ]; then
        echo "❌ Error: No available ports found for frontend"
        kill $BACKEND_PID 2>/dev/null
        exit 1
    fi
done

echo "   🌐 URL: http://localhost:$FRONTEND_PORT"

# Start frontend server
if command -v python3 &> /dev/null; then
    echo "   🔧 Starting web server (Python 3)..."
    python3 -m http.server $FRONTEND_PORT > ../frontend.log 2>&1 &
    FRONTEND_PID=$!
elif command -v python &> /dev/null; then
    echo "   🔧 Starting web server (Python)..."
    python -m http.server $FRONTEND_PORT > ../frontend.log 2>&1 &
    FRONTEND_PID=$!
else
    echo "⚠️  No Python found for frontend server."
    echo "   You can open frontend/index.html directly in your browser."
    FRONTEND_PID=""
fi

cd ..

echo ""
echo "✅ Chromatography Analysis Tool is ready!"
echo "=========================================="
echo ""
echo "🔗 Access Points:"
echo "   📊 Web Interface: http://localhost:$FRONTEND_PORT"
echo "   🔌 Backend API:   http://localhost:8080"
echo "   📖 API Docs:      http://localhost:8080/"
echo ""
echo "📝 Logs:"
echo "   🔙 Backend:  backend.log"
echo "   🔜 Frontend: frontend.log"
echo ""
echo "🛑 To stop the servers:"
echo "   Press Ctrl+C or run: kill $BACKEND_PID"
if [ -n "$FRONTEND_PID" ]; then
    echo "                        kill $FRONTEND_PID"
fi
echo ""

# Save PIDs for cleanup
echo "$BACKEND_PID" > .backend_pid
if [ -n "$FRONTEND_PID" ]; then
    echo "$FRONTEND_PID" > .frontend_pid
fi

# Try to open in browser (macOS)
if command -v open &> /dev/null; then
    echo "🌐 Opening in default browser..."
    sleep 2
    open "http://localhost:$FRONTEND_PORT"
fi

echo ""
echo "💡 Upload a chromatography image and start analyzing!"
echo "   Sample images are available in the project directory."
echo ""

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "🛑 Shutting down servers..."
    if [ -f .backend_pid ]; then
        BACKEND_PID=$(cat .backend_pid)
        kill $BACKEND_PID 2>/dev/null
        rm .backend_pid
        echo "   ✅ Backend stopped"
    fi
    if [ -f .frontend_pid ]; then
        FRONTEND_PID=$(cat .frontend_pid)
        kill $FRONTEND_PID 2>/dev/null
        rm .frontend_pid
        echo "   ✅ Frontend stopped"
    fi
    echo "👋 Goodbye!"
    exit 0
}

# Set up signal handlers
trap cleanup SIGINT SIGTERM

# Keep the script running
if [ -n "$FRONTEND_PID" ]; then
    # Wait for either process to exit
    wait $BACKEND_PID $FRONTEND_PID
else
    # Only wait for backend
    wait $BACKEND_PID
fi

cleanup
