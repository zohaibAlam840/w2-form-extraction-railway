#!/bin/bash

# Install system dependencies
echo "Installing system dependencies..."
apt-get update && apt-get install -y poppler-utils

# Install Python dependencies
echo "Installing Python dependencies..."
pip install -r requirements.txt

# Run migrations or any setup tasks if needed (optional)
# echo "Running migrations..."
# python manage.py migrate

# Additional build tasks can be added here
