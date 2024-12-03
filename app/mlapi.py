from fastapi import FastAPI, File, UploadFile, HTTPException
import pandas as pd
import joblib
from typing import Dict
import uvicorn

# Initialize FastAPI app
app = FastAPI(
    title="Sepsis Prediction API",
    description="An API for predicting sepsis using multiple machine learning models.",
    version="1.0.0",
)

# Define model paths
MODEL_PATHS = {
    "K-Nearest": "models/K-Nearest Neighbors_model.pkl",
    "LightGBM": "models/LightGBM_model.pkl",
    "Logistic Regression": "models/Logistic Regression_model.pkl",
    "Random Forest": "models/Random Forest_model.pkl",
    "Sepsis Model": "models/sepsis_model.pkl",
    "SVM": "models/SVM_model.pkl",
}

# Load models
models: Dict[str, object] = {}
for model_name, model_path in MODEL_PATHS.items():
    try:
        models[model_name] = joblib.load(model_path)
    except Exception as e:
        raise RuntimeError(
            f"Failed to load model '{model_name}' from '{model_path}'. Error: {e}"
        )

@app.get("/")
async def root():
    """
    Root endpoint to confirm the API is operational.
    """
    return {"message": "Welcome to the Sepsis Prediction API!"}

@app.post("/predict")
async def predict(model: str, file: UploadFile = File(...)):
    """
    Predict sepsis based on the provided model and uploaded CSV file.

    Args:
        model (str): Name of the model to use for prediction.
        file (UploadFile): A CSV file containing input data for prediction.

    Returns:
        dict: A dictionary containing predictions for each row in the uploaded file.
    """
    # Validate model existence
    if model not in models:
        raise HTTPException(status_code=400, detail=f"Model '{model}' not found.")

    # Process the uploaded file
    try:
        contents = await file.read()
        data = pd.read_csv(pd.compat.StringIO(contents.decode("utf-8")))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading file: {e}")

    # Generate predictions
    try:
        selected_model = models[model]
        predictions = selected_model.predict(data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")

    return {"predictions": predictions.tolist()}

# To run the app:
# uvicorn mlapi:app --reload
