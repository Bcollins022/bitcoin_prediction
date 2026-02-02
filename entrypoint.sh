#!/bin/bash
set -e

# Default to port 8080 if PORT is not set (Cloud Run sets this)
PORT=${PORT:-8080}

if [ "$SERVICE_TYPE" = "api" ]; then
    echo "Starting API on port $PORT..."
    exec uvicorn backend.api:app --host 0.0.0.0 --port $PORT
elif [ "$SERVICE_TYPE" = "frontend" ]; then
    echo "Starting Frontend on port $PORT..."
    exec streamlit run frontend/app.py --server.port $PORT --server.address 0.0.0.0
else
    echo "Error: SERVICE_TYPE env var not set to 'api' or 'frontend'."
    echo "Defaulting to API..."
    exec uvicorn backend.api:app --host 0.0.0.0 --port $PORT
fi
