# === Base pour ton backend Python ===
FROM python:3.12-slim

# Variables (pas de secret sensible ici)
ENV OL_BASE_URL=http://ollama:11434/v1

# Dépendances système
RUN apt-get update && \
    apt-get install -y git curl && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Copier l’application
WORKDIR /app
COPY app/ /app/

# Installer les dépendances Python
RUN pip install --no-cache-dir -r /app/requirements.txt

# Entrypoint
COPY ./entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 8000

CMD ["/entrypoint.sh"]