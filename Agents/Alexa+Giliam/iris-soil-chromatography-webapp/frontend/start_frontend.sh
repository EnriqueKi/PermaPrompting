#!/bin/bash

# Chromatography Analysis Frontend Startup Script

echo "🌐 Starting Chromatography Analysis Frontend..."
echo "==============================================="

# Check if we're in the right directory
if [ ! -f "index.html" ]; then
    echo "❌ Error: index.html not found. Please run this script from the frontend directory."
    exit 1
fi

# Function to check if a port is in use
port_in_use() {
    lsof -Pi :$1 -sTCP:LISTEN -t >/dev/null
}

# Find an available port starting from 3000
PORT=3000
while port_in_use $PORT; do
    PORT=$((PORT + 1))
    if [ $PORT -gt 3010 ]; then
        echo "❌ Error: No available ports found between 3000-3010"
        exit 1
    fi
done

echo "🔍 Found available port: $PORT"

# Check for different HTTP server options
if command -v python3 &> /dev/null; then
    echo "🐍 Using Python 3 HTTP server"
    echo "🚀 Frontend available at: http://localhost:$PORT"
    echo "💡 Press Ctrl+C to stop the server"
    echo ""
    python3 -m http.server $PORT
elif command -v python &> /dev/null; then
    echo "🐍 Using Python HTTP server"
    echo "🚀 Frontend available at: http://localhost:$PORT"
    echo "💡 Press Ctrl+C to stop the server"
    echo ""
    python -m http.server $PORT
elif command -v php &> /dev/null; then
    echo "🐘 Using PHP development server"
    echo "🚀 Frontend available at: http://localhost:$PORT"
    echo "💡 Press Ctrl+C to stop the server"
    echo ""
    php -S localhost:$PORT
elif command -v npx &> /dev/null; then
    echo "📦 Using Node.js serve"
    echo "🚀 Frontend available at: http://localhost:$PORT"
    echo "💡 Press Ctrl+C to stop the server"
    echo ""
    npx serve . -p $PORT
else
    echo "⚠️  No HTTP server found. You can:"
    echo "   1. Open index.html directly in your browser, or"
    echo "   2. Install one of these: Python, PHP, or Node.js"
    echo ""
    echo "💡 Direct file access:"
    echo "   file://$(pwd)/index.html"
    
    # Try to open in default browser (macOS)
    if command -v open &> /dev/null; then
        echo ""
        echo "🌐 Opening in default browser..."
        open index.html
    fi
fi
