#!/bin/bash
set -e

# === Lancer FastAPI ===
echo "🚀 Starting FastAPI server..."
exec uvicorn main:app --host 0.0.0.0 --port 8000
