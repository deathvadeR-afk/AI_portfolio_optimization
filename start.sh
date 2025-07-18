#!/bin/bash
# Portfolio Optimization API Startup Script

set -e

echo "Starting Portfolio Optimization API..."

# Set default values
export HOST=${HOST:-0.0.0.0}
export PORT=${PORT:-8000}
export WORKERS=${WORKERS:-1}
export LOG_LEVEL=${LOG_LEVEL:-info}

# Create logs directory if it doesn't exist
mkdir -p logs

# Check if models directory exists and has models
if [ ! -d "models/ppo" ]; then
    echo "Creating models directory..."
    mkdir -p models/ppo
fi

# Check if data directory exists
if [ ! -d "data" ]; then
    echo "Warning: Data directory not found. Creating empty directory..."
    mkdir -p data
fi

# Start the API server
echo "Starting API server on ${HOST}:${PORT} with ${WORKERS} workers..."

if [ "$ENVIRONMENT" = "development" ]; then
    # Development mode with auto-reload
    exec uvicorn src.api.portfolio_api:app \
        --host $HOST \
        --port $PORT \
        --log-level $LOG_LEVEL \
        --reload
else
    # Production mode with Gunicorn
    exec gunicorn src.api.portfolio_api:app \
        --bind $HOST:$PORT \
        --workers $WORKERS \
        --worker-class uvicorn.workers.UvicornWorker \
        --log-level $LOG_LEVEL \
        --access-logfile logs/access.log \
        --error-logfile logs/error.log \
        --capture-output \
        --enable-stdio-inheritance
fi
