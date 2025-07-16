import streamlit as st
import requests
from PIL import Image
import io
import os

st.set_page_config(page_title="Fruits 360 Classifier", layout="centered")
st.title("🍎 Fruits 360 Classifier")
st.write("Uploadez une image de fruit pour obtenir la prédiction du modèle.")

# Utiliser la variable d'environnement API_URL si présente (pour docker-compose)
API_URL = os.getenv("API_URL", "http://localhost:8000")

# Drag and drop uploader
uploaded_file = st.file_uploader("Glissez-déposez une image ici", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Image uploadée", use_column_width=True)
    st.write("Prédiction en cours...")

    # Préparer le fichier pour l'API
    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
    try:
        response = requests.post(f"{API_URL}/predict", files=files)
        if response.status_code == 200:
            result = response.json()
            st.success(f"Classe prédite : {result['predicted_class']}")
            st.info(f"Confiance : {result['confidence']*100:.2f}%")
        else:
            st.error(f"Erreur API : {response.status_code} - {response.text}")
    except Exception as e:
        st.error(f"Erreur lors de la requête à l'API : {e}")
else:
    st.info("Veuillez uploader une image au format JPG ou PNG.") 