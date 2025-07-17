from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = FastAPI(title="API Fruits360", version="1.0")

# Charger le modèle sauvegardé
model = tf.keras.models.load_model("resnet50_fruits360.h5")  # adapte le chemin

# Liste des classes (ex. Fruits360, à adapter)
class_names = ["Apple", "Banana", "Cherry", "Kiwi", "Mango", "Orange", "Strawberry"]

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize((100, 100))  # ⚠️ .convert("RGB")
        image_array = np.expand_dims(np.array(image) / 255.0, axis=0)

        prediction = model.predict(image_array)
        predicted_class = class_names[np.argmax(prediction)]
        confidence = float(np.max(prediction))

        return JSONResponse(content={"class": predicted_class, "confidence": confidence})

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})