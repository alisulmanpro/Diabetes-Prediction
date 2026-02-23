from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
from pathlib import Path

MODEL_PATH = Path(__file__).resolve().parents[0] / "models" / "xgb_model.pkl"

app = FastAPI(title="Diabetes Prediction API")

model = joblib.load(MODEL_PATH)

# Define input schema
class PatientData(BaseModel):
    Pregnancies: int
    Glucose: float
    BloodPressure: float
    SkinThickness: float
    Insulin: float
    BMI: float
    DiabetesPedigreeFunction: float
    Age: int

@app.get("/")
def home():
    return {"message": "Diabetes Prediction API is running"}

@app.post("/predict")
def predict(data: PatientData):
    input_data = np.array([[
        data.Pregnancies,
        data.Glucose,
        data.BloodPressure,
        data.SkinThickness,
        data.Insulin,
        data.BMI,
        data.DiabetesPedigreeFunction,
        data.Age
    ]])

    probability = model.predict_proba(input_data)[0][1]
    prediction = int(probability >= 0.40)

    return {
        "diabetes_probability": float(round(probability, 4)),
        "prediction": prediction
    }
