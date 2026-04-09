#!/bin/bash
set -e
cd api

echo "Starting Gunicorn on port ${PORT:-8080}..."
exec gunicorn --bind 0.0.0.0:${PORT:-8080} \
     --workers 1 \
     --threads 4 \
     --timeout 120 \
     --access-logfile - \
     --error-logfile - \
     parse:app
