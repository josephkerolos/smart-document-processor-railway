#!/bin/bash

# Start script for Smart Document Processor on Railway

echo "Starting Smart Document Processor..."

# Railway provides PORT - we'll use it for the main app
export PORT=${PORT:-8080}

# Log environment
echo "PORT: $PORT"
echo "GEMINI_API_KEY: ${GEMINI_API_KEY:0:10}..." # Show first 10 chars only

# For Railway, we'll run everything on the single PORT
# Start the backend API server on the Railway-provided PORT
echo "Starting backend server on port $PORT..."
cd /app/backend
python3 -m uvicorn main_enhanced_v12:app \
    --host 0.0.0.0 \
    --port $PORT \
    --workers 1 \
    --log-level info &

BACKEND_PID=$!

# Keep the process running
wait $BACKEND_PID