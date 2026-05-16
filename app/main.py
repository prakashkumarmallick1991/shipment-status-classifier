from fastapi import FastAPI
from pydantic import BaseModel
import joblib


# -----------------------------
# Load Saved Files
# -----------------------------

model = joblib.load("model.pkl")

vectorizer = joblib.load("vectorizer.pkl")


# -----------------------------
# Initialize FastAPI
# -----------------------------

app = FastAPI(
    title="Shipment Status Classifier API",
    description="Predict Internal Shipment Status",
    version="1.0"
)


# -----------------------------
# Request Schema
# -----------------------------

class StatusRequest(BaseModel):
    text: str


# -----------------------------
# Root Endpoint
# -----------------------------

@app.get("/")
def home():

    return {
        "message": "Shipment Status Classifier API Running"
    }


# -----------------------------
# Prediction Endpoint
# -----------------------------

@app.post("/predict")
def predict_status(request: StatusRequest):

    input_text = request.text

    # transform input text
    text_vector = vectorizer.transform([input_text])

    # get prediction probabilities
    probabilities = model.predict_proba(text_vector)

    # highest confidence score
    confidence = probabilities.max()

    # predicted label
    prediction = model.predict(text_vector)[0]

    return {
        "external_status": input_text,
        "predicted_internal_status": prediction,
        "confidence": round(float(confidence), 2)
    }