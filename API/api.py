from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from PIL import Image
import io
import os

app = FastAPI(
    title="Image Classification API",
    description="API pour classifier des images avec un modèle Deep Learning entraîné",
    version="1.0.0",
    docs_url=None, redoc_url=None, openapi_url=None  # Désactive la doc auto
)

# Charger dynamiquement les noms de classes depuis le dossier d'entraînement
TRAIN_DIR = "fruits360_data/Training"
def get_class_names(train_dir):
    try:
        return sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])
    except Exception as e:
        raise RuntimeError(f"Erreur lors du chargement des classes : {e}")

# Charger le modèle au démarrage
try:
    model = load_model("resnet50_fruits360.h5")
    class_names = get_class_names(TRAIN_DIR)
except Exception as e:
    raise RuntimeError(f"Erreur lors du chargement du modèle ou des classes : {e}")

# Prétraitement des images (adapté à ResNet50 entraîné sur Fruits 360)
def preprocess_image(image: Image.Image) -> np.ndarray:
    image = image.resize((100, 100))  # Adapter à l'entrée du modèle
    image = img_to_array(image)
    image = image / 255.0  # Normalisation
    image = np.expand_dims(image, axis=0)
    return image

@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    if file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(status_code=400, detail="Format d'image non supporté (JPEG/PNG uniquement)")

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        image = preprocess_image(image)
        prediction = model.predict(image)
        predicted_class = class_names[np.argmax(prediction[0])]
        confidence = float(np.max(prediction[0]))
        return JSONResponse(content={
            "predicted_class": predicted_class,
            "confidence": round(confidence, 4)
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la prédiction : {str(e)}")
