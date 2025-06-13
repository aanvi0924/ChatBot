# Use official PyTorch image with CUDA (for GPU support)
FROM python:3.11
WORKDIR /app
RUN pip install pip==21.3.1


# Set environment variables
ENV PYTHONUNBUFFERED=1

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y nano procps wget tar curl unzip sqlite3 build-essential libsqlite3-dev ca-certificates && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*



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
