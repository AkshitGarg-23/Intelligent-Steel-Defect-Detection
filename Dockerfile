# Use standard lightweight Python base image
FROM python:3.11-slim

# Prevent Python from writing .pyc files to disk and enable line buffering for logging
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set the working directory
WORKDIR /app

# Install basic compile-time utilities needed for some wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies first to leverage Docker caching
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application files
COPY . /app/

# Expose port 8000 for FastAPI
EXPOSE 8000

# Set entrypoint to run FastAPI via uvicorn
CMD ["python", "-m", "api.main"]
