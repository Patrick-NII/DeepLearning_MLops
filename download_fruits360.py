import os
import zipfile
from kaggle.api.kaggle_api_extended import KaggleApi

# Utiliser le dossier .kaggle local pour la config
os.environ['KAGGLE_CONFIG_DIR'] = os.path.join(os.path.dirname(__file__), '.kaggle')

DATASET = 'moltean/fruits'
OUTPUT_DIR = 'fruits360_data'
ZIP_PATH = os.path.join(OUTPUT_DIR, 'fruits.zip')

# Créer le dossier de sortie si besoin
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Initialiser l'API Kaggle
api = KaggleApi()
api.authenticate()

# Télécharger le dataset
print('Téléchargement du dataset Fruits 360...')
api.dataset_download_files(DATASET, path=OUTPUT_DIR, unzip=False)
print('Téléchargement terminé.')

# Dézipper le fichier téléchargé
print('Décompression...')
with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
    zip_ref.extractall(OUTPUT_DIR)
print('Décompression terminée.') 