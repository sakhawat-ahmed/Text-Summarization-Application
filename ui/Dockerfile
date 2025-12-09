# ui/Dockerfile
FROM python:3.12-slim

WORKDIR /app

# Copy UI source code
COPY . /app

# Copy local packages
COPY packages ./packages

# Install pip from local wheel (optional)
# RUN pip install --no-cache-dir ./packages/pip-25.3-py3-none-any.whl

# Install all packages offline
RUN pip install --no-cache-dir ./packages/*

# Expose UI port (adjust if needed)
EXPOSE 3000

# Command to run UI
CMD ["python", "app.py"]
