# Use official PyTorch image with CUDA (for GPU support)
FROM python:3.10-slim-bullseye


# Set environment variables
ENV PYTHONUNBUFFERED=1

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git curl nano \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip first
RUN pip install --upgrade pip

# Copy only requirements first (use Docker cache)
COPY requirement_rag.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirement_rag.txt

# Copy the rest of the application
COPY . .

# Expose Flask default port
EXPOSE 5000

# Run the Flask app
CMD ["python", "rag_server.py"]
