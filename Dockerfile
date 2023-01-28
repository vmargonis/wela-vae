FROM python:3.9-slim-buster

WORKDIR /app

# Necessary dependencies for openCV
RUN apt-get update \
    && apt-get install -y libgl1-mesa-glx libglib2.0-0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY requirements/base.txt /app/requirements/base.txt

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r /app/requirements/base.txt

COPY lib/ /app/lib/
