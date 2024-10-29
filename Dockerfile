FROM python:3.10-slim-buster
FROM python:3.10-slim-buster

# Install necessary packages including OpenCV dependencies
RUN apt update -y && \
    apt install -y awscli libgl1-mesa-glx && \
    rm -rf /var/lib/apt/lists/*  # Clean up to reduce image size

WORKDIR /app

COPY . /app
RUN pip install -r requirements.txt

# Command to run the application
CMD ["python3", "app.py"]
