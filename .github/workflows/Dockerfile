# === Base ===
FROM python:3.12-slim

# === Variables ===
ENV OL_BASE_URL=http://localhost:11434/v1
ENV OL_API_KEY=ollama

# === Dépendances système ===
RUN apt-get update && \
    apt-get install -y curl git wget unzip && \
    apt-get clean

# === Ollama CLI ===
RUN curl -sSL https://ollama.com/install.sh | bash

# === Copy Python app ===
WORKDIR /app
COPY app/ /app/

# === Python dependencies ===
RUN pip install --no-cache-dir -r requirements.txt

# === Entrypoint ===
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 8000 11434

CMD ["/entrypoint.sh"]