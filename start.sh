#!/bin/bash

# Start script for Smart Document Processor on Railway

echo "Starting Smart Document Processor..."

# Set default values if not provided by Railway
export PORT=${PORT:-8080}
export BACKEND_PORT=${BACKEND_PORT:-4830}

# Log environment
echo "PORT: $PORT"
echo "BACKEND_PORT: $BACKEND_PORT"
echo "GEMINI_API_KEY: ${GEMINI_API_KEY:0:10}..." # Show first 10 chars only

# Start backend API server
echo "Starting backend server on port $BACKEND_PORT..."
cd /app/backend
python3 -m uvicorn main_enhanced_v12:app \
    --host 0.0.0.0 \
    --port $BACKEND_PORT \
    --workers 2 \
    --log-level info &

BACKEND_PID=$!

# Wait for backend to be ready
echo "Waiting for backend to start..."
sleep 5

# Check if backend is running
if ! kill -0 $BACKEND_PID 2>/dev/null; then
    echo "Backend failed to start!"
    exit 1
fi

# Serve the frontend (static files) using Python's http.server
echo "Starting frontend server on port $PORT..."
cd /app/frontend/build
python3 -m http.server $PORT --bind 0.0.0.0 &

FRONTEND_PID=$!

# Function to handle shutdown
shutdown() {
    echo "Shutting down..."
    kill $BACKEND_PID $FRONTEND_PID 2>/dev/null
    exit 0
}

# Set up signal handlers
trap shutdown SIGINT SIGTERM

# Keep the script running and monitor processes
while true; do
    # Check if both processes are still running
    if ! kill -0 $BACKEND_PID 2>/dev/null; then
        echo "Backend process died, restarting..."
        cd /app/backend
        python3 -m uvicorn main_enhanced_v12:app \
            --host 0.0.0.0 \
            --port $BACKEND_PORT \
            --workers 2 \
            --log-level info &
        BACKEND_PID=$!
    fi
    
    if ! kill -0 $FRONTEND_PID 2>/dev/null; then
        echo "Frontend process died, restarting..."
        cd /app/frontend/build
        python3 -m http.server $PORT --bind 0.0.0.0 &
        FRONTEND_PID=$!
    fi
    
    sleep 30
done