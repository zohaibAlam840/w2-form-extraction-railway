# Use an official Python runtime as a parent image
FROM python:3.12.7-slim as build-stage

# Set environment variables to avoid interaction
ENV DEBIAN_FRONTEND=noninteractive

# Install dependencies
RUN apt-get update && apt-get install -y \
    poppler-utils \
    libpoppler-cpp-dev \
    gcc \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /main

# Copy the current directory contents into the container
COPY ./ main

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "${PORT:-8000}"]
