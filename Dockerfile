# Use official Python runtime as base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Set environment variables
ENV FLASK_APP=appgrok14.py
ENV FLASK_RUN_HOST=0.0.0.0
# GROQ_API_KEY will be set via Render environment variables

# Expose port
EXPOSE 8000

# Run the application with gunicorn for production (optional, uncomment for production)
# CMD ["gunicorn", "--bind", "0.0.0.0:8000", "appgrok14:app"]

# Run the application directly (development mode)
CMD ["python", "appgrok14.py"]