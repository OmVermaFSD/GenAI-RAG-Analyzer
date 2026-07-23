# Multi-stage production Dockerfile for DocuMind AI – RAG Knowledge Retrieval Platform
FROM python:3.10-slim

WORKDIR /app

# Set unbuffered output for real-time log streaming
ENV PYTHONUNBUFFERED=1

# Install system dependencies (build tools & curl)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy and install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY app/ ./app/
COPY app.py .
COPY .env.example .env

# Create data storage directories
RUN mkdir -p data/uploads data/chroma_db

# Expose FastAPI port
EXPOSE 8000

# Run Uvicorn ASGI Server
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
