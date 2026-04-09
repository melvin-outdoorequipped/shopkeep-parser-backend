#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# 1. Move into the directory containing your Flask app
# (Only needed if your 'api' folder isn't the root of the repo)
cd api

# 2. Install dependencies 
# Note: Railway usually does this during the build phase, 
# but this ensures they are present in the runtime.
pip install --no-cache-dir -r requirements.txt

# 3. Launch Gunicorn
# 'parse:app' assumes your file is 'parse.py' and your Flask instance is 'app'
echo "Starting Gunicorn on port ${PORT:-8080}..."

exec gunicorn --bind 0.0.0.0:${PORT:-8080} \
     --workers 4 \
     --timeout 120 \
     --access-logfile - \
     --error-logfile - \
     parse:app
