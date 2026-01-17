# Use an official Python runtime as a parent image
FROM python:3.12-slim

# Set work directory
WORKDIR /app

# Install system dependencies (needed for TA-Lib and other libraries)
# build-essential and python3-dev for compiling C extensions
# curl for healthchecks
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    curl \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install TA-Lib C library
# Download and install TA-Lib C library
RUN curl -L http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz -o ta-lib-0.4.0-src.tar.gz && \
    tar -xzf ta-lib-0.4.0-src.tar.gz && \
    cd ta-lib && \
    ./configure --prefix=/usr && \
    make && \
    make install && \
    cd .. && \
    rm -rf ta-lib ta-lib-0.4.0-src.tar.gz

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
# Update requirements to not use ta-lib-binary if it fails on linux, we use standard ta-lib since we installed C lib
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Create output directory for persistence
RUN mkdir -p output/models/lightgbm

# Expose port
EXPOSE 8000

# Run the API
CMD ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
