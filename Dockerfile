# Use an official Python runtime as a parent image
FROM python:3.12-bullseye

# Set the working directory in the container
WORKDIR /app

# Install system dependencies, including ffmpeg, and clean up
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy the dependencies file to the working directory
COPY requirements.txt .

# Install Python dependencies
# This will install the CPU version of PyTorch by default
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application's code to the working directory
COPY . .

# Create directories for output files.
# These paths will be targeted by the docker-compose volumes.
RUN mkdir -p /app/downloads /app/separated /app/karaoke

# Make port 5001 available to the world outside this container
EXPOSE 5001

# Define environment variables for Flask
ENV FLASK_APP=app.py
ENV FLASK_RUN_HOST=0.0.0.0

# Run app.py when the container launches
CMD ["python", "app.py"]
