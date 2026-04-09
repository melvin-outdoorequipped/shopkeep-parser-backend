#!/bin/bash

# Exit on any error
set -e

# Move into the app folder where parse.py lives
cd api

# Launch Gunicorn (pip install is removed because Nixpacks did it during build)
echo "Starting Gunicorn on port ${PORT:-8080}..."
exec gunicorn --bind 0.0.0.0:${PORT:-8080} \
     --workers 4 \
     --timeout 120 \
     --access-logfile - \
     --error-logfile - \
     parse:app
