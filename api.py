
from fastapi import FastAPI, UploadFile, File, HTTPException
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

app = FastAPI(title="Plant Species Detection API")

model = tf.keras.models.load_model(
    "/content/drive/MyDrive/CP-1/improved_plant_model.keras"
)

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
    return {"message": "Plant Species Detection API is running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    import io
    from PIL import UnidentifiedImageError

    try:
        raw_bytes = await file.read()
        if not raw_bytes:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        try:
            image = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
        except (UnidentifiedImageError, OSError, ValueError):
            raise HTTPException(
                status_code=400,
                detail="Invalid image. Please upload a JPG, JPEG, or PNG image."
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
        low_confidence = confidence < 50.0

        return {
            "predicted_plant": top_predictions[0]["plant"],
            "confidence": confidence,
            "top_3_predictions": top_predictions,
            "low_confidence_warning": low_confidence,
            "message": (
                "Low confidence: please upload a clearer image of a supported plant."
                if low_confidence
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

