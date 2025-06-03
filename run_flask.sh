#!/bin/bash

# Pull the latest model using DVC
dvc pull models/model.pkl

# Create a deployment folder (if required)
if [ ! -d "deployed" ]; then
    mkdir deployed
fi

# Export MLFlow tracking URI
export MLFLOW_TRACKING_URI=mlruns

# Copy necessary files to the deployment folder
cp -r app/templates/ deployed/templates/
cp app/app.py deployed/
cp models/model.pkl deployed/

# Navigate to deployment directory
cd deployed

# Ensure MLflow server is running
if ! pgrep -f "mlflow.server" > /dev/null; then
    echo "MLflow server not running. Starting it..."
    mlflow server \
        --backend-store-uri sqlite:///mlflow.db \
        --default-artifact-root ./mlruns \
        --host 127.0.0.1 \
        --port 5000 &
    sleep 5  # Wait for the server to start
else
    echo "MLflow server is already running."
fi

# # Start the Flask app in the background
# nohup python app.py &

# # Now the Flask app is running in the background, and the script continues
# # Perform any additional tasks, such as logging, metrics collection, etc.

# echo "Flask app is running in the background."
