# === Base pour ton backend Python ===
ARG PYTHON_VERSION=3.12
ARG BUILD_VERSION=dev
FROM python:${PYTHON_VERSION}-slim

# Variables (pas de secret sensible ici)
ENV OL_BASE_URL=http://51.68.44.166:11434/api/chat

LABEL org.opencontainers.image.version=${BUILD_VERSION}

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