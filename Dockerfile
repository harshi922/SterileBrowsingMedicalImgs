# Start from Python 3.10 slim image
FROM python:3.10-slim-buster

# Update and install necessary system libraries
RUN apt update -y && \
    apt install -y --no-install-recommends \
        awscli \
        libgl1-mesa-glx \
        libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy application code
COPY . /app

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Define the command to run the app
CMD ["python3", "app.py"]
