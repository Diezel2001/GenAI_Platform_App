#!/bin/bash

# Activate the virtual environment
source genai_venv/bin/activate

docker run -d --name redis-stack -p 6379:6379 redis/redis-stack:latest

cd backend
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# metrics
docker run -d \
  -p 9090:9090 \
  -v $(pwd)/prometheus.yml:/etc/prometheus/prometheus.yml \
  prom/prometheus

docker run -d -p 3000:3000 grafana/grafana
docker run -d -p 3000:3000 langfuse/langfuse