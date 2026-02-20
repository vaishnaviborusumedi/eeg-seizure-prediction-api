from fastapi import FastAPI

from pydantic import BaseModel

import joblib

import numpy as np

import sklearn


# STEP 1: Create FastAPI app

app = FastAPI(title="EEG Seizure Prediction API", version="1.0")


# STEP 2: Load model

model = joblib.load("model.pkl")

print("Model loaded successfully")


# STEP 3: Input schema

class EEGInput(BaseModel):

    X1: float
    X2: float
    X3: float
    X4: float
    X5: float
    X6: float
    X7: float
    X8: float
    X9: float
    X10: float
    X11: float
    X12: float
    X13: float
    X14: float
    X15: float
    X16: float


# STEP 4: Health check endpoint

@app.get("/")

def health_check():

    return {

        "status": "API is running",

        "model_loaded": True

    }


# STEP 5: Model info endpoint

@app.get("/model-info")

def model_info():

    return {

        "model_type": str(type(model.named_steps["model"]).__name__),

        "sklearn_version": sklearn.__version__

    }


# STEP 6: Single prediction endpoint with confidence score

@app.post("/predict")

def predict(data: EEGInput):

    input_data = np.array([[

        data.X1,
        data.X2,
        data.X3,
        data.X4,
        data.X5,
        data.X6,
        data.X7,
        data.X8,
        data.X9,
        data.X10,
        data.X11,
        data.X12,
        data.X13,
        data.X14,
        data.X15,
        data.X16

    ]])

    prediction = model.predict(input_data)[0]

    probability = model.predict_proba(input_data)[0]

    confidence = float(max(probability))


    if prediction == 1:
        result = "Seizure Risk Detected"
    else:
        result = "No Seizure Risk"


    return {

        "prediction": int(prediction),

        "result": result,

        "confidence_score": round(confidence, 4)

    }


# STEP 7: Batch prediction endpoint

@app.post("/predict-batch")

def predict_batch(data: list[EEGInput]):

    input_list = []

    for item in data:

        input_list.append([

            item.X1,
            item.X2,
            item.X3,
            item.X4,
            item.X5,
            item.X6,
            item.X7,
            item.X8,
            item.X9,
            item.X10,
            item.X11,
            item.X12,
            item.X13,
            item.X14,
            item.X15,
            item.X16

        ])

    input_array = np.array(input_list)

    predictions = model.predict(input_array)

    probabilities = model.predict_proba(input_array)

    results = []

    for i in range(len(predictions)):

        results.append({

            "prediction": int(predictions[i]),

            "confidence": float(max(probabilities[i]))

        })

    return {

        "batch_predictions": results

    }