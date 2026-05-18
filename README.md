# RAGService — OLAP MDX Generation API

## Description

RAGService est un backend FastAPI qui convertit des requêtes métier en requêtes MDX à l'aide d'un système RAG (Retrieval-Augmented Generation). Il indexe les métadonnées OLAP depuis un fichier JSON, génère un prompt enrichi et envoie la requête à un modèle LLM (Ollama) pour produire du MDX.

## Architecture

- `app/main.py`: point d'entrée FastAPI.
- `app/rag_service_olap.py`: service RAG principal utilisant FAISS + `sentence-transformers` pour l'indexation et la recherche.
- `app/RAG_chroma_service.py`: variante Chroma de RAG présente dans le dépôt, mais non utilisée par le service principal.
- `app/data/metadata.json`: définition du cube OLAP, mesures, dimensions et exemples de requêtes.
- `Dockerfile`: build Docker du backend.
- `docker-compose.yml`: configuration local pour exécuter le service backend.
- `.github/workflows/cd.yml`: pipeline CI/CD GitHub Actions.

## Structure du dépôt

```
.
├─ .github/workflows/cd.yml
├─ Dockerfile
├─ docker-compose.yml
├─ entrypoint.sh
├─ requirements.txt
├─ README.md
├─ app/
│  ├─ __init__.py
│  ├─ main.py
│  ├─ rag_service_olap.py
│  ├─ RAG_chroma_service.py
│  ├─ requirements.txt
│  └─ data/metadata.json
└─ tests/
   └─ test_smoke.py
```

## Prérequis

- Python 3.12
- Docker
- Docker Compose (optionnel pour exécution locale)

## Installation locale

1. Depuis la racine du projet, installer les dépendances :

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

2. Lancer le service depuis le dossier `app` (nécessaire car `main.py` charge `./data/metadata.json`) :

```bash
cd app
uvicorn main:app --host 0.0.0.0 --port 8000
```

3. Appeler l’API :

```bash
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Quel MDX pour le chiffre d'affaires par année ?"}'
```

## Configuration Docker

### Build

```bash
docker build -t ragservice .
```

### Run

```bash
docker run -p 8000:8000 \
  -e OL_BASE_URL=http://<OLLAMA_HOST>:11434/api/chat \
  ragservice
```

> Note : `entrypoint.sh` démarre uniquement `uvicorn`. Le service Ollama doit être accessible séparément via `OL_BASE_URL`.

## Docker Compose

Pour lancer le service backend avec Docker Compose :

```bash
docker compose up --build
```

Le service sera exposé sur `http://localhost:8000`.

## Variables d’environnement

- `OL_BASE_URL`: URL de l’API Ollama (par défaut `http://51.68.44.166:11434/api/chat`).

## Tests

Le dépôt contient un test de fumée simple dans `tests/test_smoke.py`.

Exécution des tests :

```bash
pytest
```

## CI/CD GitHub Actions

Le pipeline `.github/workflows/cd.yml` effectue :

1. Lint et tests Python (`black`, `isort`, `flake8`, `pytest`)
2. Build et push Docker vers GHCR
3. Déploiement SSH vers un VPS si le push est sur `main`
4. Notifications Telegram en cas de succès ou d’échec

### Secrets requis pour le déploiement

- `VPS_HOST`
- `VPS_SSH_KEY`
- `VPS_USER`
- `GHCR_TOKEN`
- `TELEGRAM_TOKEN`
- `TELEGRAM_CHAT_ID`

## Points importants

- Le backend charge le cube OLAP à partir de `app/data/metadata.json` au démarrage.
- La recherche s’appuie sur une indexation FAISS et `sentence-transformers`.
- La génération MDX ne se fait pas localement dans le code : le service appelle un endpoint Ollama.
- Si vous souhaitez exécuter le projet localement, assurez-vous que `OLLAMA` est disponible et joignable.

## Améliorations possibles

- Ajouter des tests d’intégration pour l’API `/ask`
- Ajouter un `docker-compose` complet avec Ollama et backend
- Externaliser la configuration dans un fichier `.env`
- Installer un suivi de couverture et de logs pour la production
