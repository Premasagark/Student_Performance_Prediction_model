from fastapi import FastAPI
from schemas import ModelInput
import pickle
import numpy as np


app = FastAPI(title="AI-Based Student Performance Prediction System")

# Load trained model
with open("model/model.pkl", "rb") as f:
    saved_objects = pickle.load(f)

model = saved_objects["model"]
scaler = saved_objects["scaler"]
label_encoder = saved_objects["label_encoder"]




@app.get("/")
def root():
    return {
        "message": "AI-Based Student Performance Prediction System",
        "endpoints": {
            "/predict": "POST - Predict student performance",
            "/docs": "Use Swagger UI for validate /predict"
        }
    }



# Prediction endpoint
@app.post("/predict")
def predict_performance(data: ModelInput):

    # Convert input to array
    array_data = np.array([[data.study_hours,data.attendance,data.internal_marks,data.practice_hours]])


    # Scale input
    input_scaled = scaler.transform(array_data)

    # Predict class
    prediction_encoded = model.predict(input_scaled)[0]
    prediction_label = label_encoder.inverse_transform([prediction_encoded])[0]

    # Confidence score
    probabilities = model.predict_proba(input_scaled)[0]
    confidence = max(probabilities) * 100

    return {
        "prediction": prediction_label,
        "confidence": f"{confidence:.0f}%"
    }
