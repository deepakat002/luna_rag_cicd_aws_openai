FROM python:3.10-slim

# Environment setup
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# RUN pip install torch==2.6.0 --extra-index-url https://download.pytorch.org/whl/cu124


# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

#Copy application files
# COPY . .

# Expose Chainlit default port
EXPOSE 8000
