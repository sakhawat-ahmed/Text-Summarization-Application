# Use Python 3.12 slim
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Copy package dependencies first (for caching pip install)
COPY packages ./packages

# Install Python dependencies
RUN pip install --no-cache-dir ./packages/*

# Copy the backend source code
COPY ./local_models /app/local_models

# Expose backend API port
EXPOSE 8000

# Default command to run the backend
CMD ["python3", "main.py"]
