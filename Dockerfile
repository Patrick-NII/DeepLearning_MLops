# Image de base ARM64 compatible avec Apple Silicon
FROM --platform=linux/amd64 python:3.10-slim

# Install dependencies système
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copie de tous les fichiers
COPY . .

# Installation des dépendances Python
RUN pip install --no-cache-dir -r requirements.txt

# Port FastAPI
EXPOSE 8000

# Lancement de l’API
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]