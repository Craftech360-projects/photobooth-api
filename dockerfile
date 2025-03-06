# Use Python 3.9 slim base image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    wget \
    libgl1 \
    libglib2.0-0 \
    build-essential \
    g++ \
    cmake && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Create necessary directories
RUN mkdir -p models uploads results
    chmod 777 /app/uploads /app/results

# Download required models during build
RUN wget -O models/inswapper_128.onnx https://huggingface.co/ezioruan/inswapper_128.onnx/resolve/main/inswapper_128.onnx && \
    wget -O models/GFPGANv1.4.pth https://huggingface.co/gmk123/GFPGAN/resolve/main/GFPGANv1.4.pth

# Copy application code
COPY . .

# Expose the application port
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]