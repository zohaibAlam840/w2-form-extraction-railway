# Start from the official Python 3.9 slim image
FROM python:3.9-slim

# Set environment variables to avoid interaction during installation
ENV DEBIAN_FRONTEND=noninteractive

# Install required system dependencies
RUN apt-get update && apt-get install -y \
    poppler-utils \
    libpoppler-cpp-dev \
    gcc \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory inside the container
WORKDIR /main

# Copy the requirements.txt file first to leverage Docker caching
COPY requirements.txt /main/

# Upgrade pip and install the Python dependencies
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . /main

# Expose port 8000 for FastAPI
# Expose port for FastAPI
EXPOSE 8000

# Command to run the application
CMD uvicorn main:app
