#!/bin/bash
set -e

# === Lancer Ollama server en arrière-plan ===
echo "🚀 Starting Ollama server..."
ollama serve &

# === Attendre que Ollama soit prêt ===
sleep 10

# === Lancer FastAPI ===
echo "🚀 Starting FastAPI server..."
uvicorn main:app --host 0.0.0.0 --port 8000