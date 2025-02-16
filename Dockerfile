FROM python:3.9-slim

WORKDIR /app

# Install OpenGL libraries and other dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements/base.txt /app/requirements/base.txt

# Install Python dependencies
RUN pip install --upgrade pip && pip install -r /app/requirements/base.txt

# Copy the rest of the application files
COPY lib/ /app/lib/
