import os
import io
import numpy as np
import tensorflow as tf

from fastapi import FastAPI, UploadFile, File, HTTPException
from PIL import Image, UnidentifiedImageError
from huggingface_hub import hf_hub_download
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

app = FastAPI(title="Plant Species Detection API")

# Download the trained model from Hugging Face
MODEL_PATH = hf_hub_download(
    repo_id="khanshadabifhe/plant-species-detection-model",
    filename="improved_plant_model.keras",
    repo_type="model"
)

# Load the trained model
model = tf.keras.models.load_model(MODEL_PATH)

# The 30 classes used during training
class_names = [
    "aloevera", "banana", "bilimbi", "cantaloupe", "cassava",
    "coconut", "corn", "cucumber", "curcuma", "eggplant",
    "galangal", "ginger", "guava", "kale", "longbeans",
    "mango", "melon", "orange", "paddy", "papaya",
    "peperchili", "pineapple", "pomelo", "shallot", "soybeans",
    "spinach", "sweetpotatoes", "tobacco", "waterapple", "watermelon"
]


@app.get("/")
def home():
    return {
        "message": "Plant Species Detection API is running"
    }


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        raw_bytes = await file.read()

        if not raw_bytes:
            raise HTTPException(
                status_code=400,
                detail="Uploaded file is empty."
            )

        try:
            image = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
        except (UnidentifiedImageError, OSError, ValueError):
            raise HTTPException(
                status_code=400,
                detail="Invalid image. Please upload a JPG, JPEG, or PNG."
            )

        image = image.resize((224, 224))
        image_array = np.array(image, dtype=np.float32)
        image_array = np.expand_dims(image_array, axis=0)
        image_array = preprocess_input(image_array)

        predictions = model.predict(image_array, verbose=0)[0]
        top_indices = np.argsort(predictions)[-3:][::-1]

        top_predictions = [
            {
                "plant": class_names[int(index)],
                "confidence": float(predictions[index] * 100)
            }
            for index in top_indices
        ]

        confidence = top_predictions[0]["confidence"]

        return {
            "predicted_plant": top_predictions[0]["plant"],
            "confidence": confidence,
            "top_3_predictions": top_predictions,
            "low_confidence_warning": confidence < 50.0,
            "message": (
                "Low confidence: upload a clearer image of a supported plant."
                if confidence < 50.0
                else "Prediction completed."
            )
        }

    except HTTPException:
        raise

    except Exception:
        raise HTTPException(
            status_code=500,
            detail="Prediction failed. Please try another image."
        )

