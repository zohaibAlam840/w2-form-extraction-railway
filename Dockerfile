FROM python:3.9-slim

ENV DEBIAN_FRONTEND=noninteractive

# Install dependencies for pdf2image, poppler, and other libraries
RUN apt-get update && apt-get install -y \
    poppler-utils \
    libpoppler-cpp-dev \
    gcc \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /main

# Copy the requirements file and install dependencies
COPY requirements.txt /main/
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code
COPY . /main

EXPOSE 8000

# Use $PORT environment variable instead of hardcoding 8000
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT}"]
