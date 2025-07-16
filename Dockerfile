# Utilise une image Python officielle
FROM python:3.9-slim-buster

# Définir le répertoire de travail
WORKDIR /app

# Copier les fichiers de dépendances
COPY requirements.txt ./

# Installer les dépendances
RUN pip install --no-cache-dir -r requirements.txt

# Copier le reste du code
COPY . .

# Exposer le port de l'API
EXPOSE 8000

# Lancer l’API avec Uvicorn
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]

# Pour lancer le front Streamlit (en local ou dans le conteneur) :
# streamlit run app_streamlit.py --server.port 8501 --server.address 0.0.0.0