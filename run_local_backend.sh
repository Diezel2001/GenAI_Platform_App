#!/bin/bash

# Activate the virtual environment
source genai_venv/bin/activate

docker run -d --name redis-stack -p 6379:6379 redis/redis-stack:latest

cd backend
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
