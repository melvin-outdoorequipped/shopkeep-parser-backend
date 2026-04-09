#!/bin/bash

# Move into the app folder
cd api

# Install dependencies
pip install -r requirements.txt

# Run the app
# For Flask:
# export FLASK_APP=main.py
# flask run --host=0.0.0.0 --port=$PORT

# For FastAPI with Uvicorn:
uvicorn main:app --host 0.0.0.0 --port=$PORT
