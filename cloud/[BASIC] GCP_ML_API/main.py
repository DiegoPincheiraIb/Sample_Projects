# To deploy, run the following command:
# $ gcloud run deploy iris-api --source . --region us-central1 --allow-unauthenticated

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import numpy as np
import joblib
import os


app = FastAPI(
    title      = "Iris Classifier — Cloud Run",
    description= "XGBoost iris model deployed on Google Cloud Run.",
    version    = "1.0.0",
)

CLASS_NAMES = ["setosa", "versicolor", "virginica"]
_model = None


def get_model():
    global _model
    if _model is None:
        model_path = os.environ.get("MODEL_PATH", "iris_model.pkl")
        if not os.path.exists(model_path):
            raise RuntimeError(f"Model not found at '{model_path}'. Run train_model.py first.")
        _model = joblib.load(model_path)
    return _model


class IrisFeatures(BaseModel):
    sepal_length: float = Field(..., ge=0, example=5.1)
    sepal_width:  float = Field(..., ge=0, example=3.5)
    petal_length: float = Field(..., ge=0, example=1.4)
    petal_width:  float = Field(..., ge=0, example=0.2)


class PredictionResponse(BaseModel):
    predicted_class: str
    class_index: int
    probabilities: dict[str, float]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(features: IrisFeatures):
    """
    Predict the iris class based on sepal and petal measurements.
    """
    model = get_model()
    X = np.array([[features.sepal_length, features.sepal_width,
                   features.petal_length, features.petal_width]])
    try:
        idx   = int(model.predict(X)[0])
        probs = model.predict_proba(X)[0]
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return PredictionResponse(
        predicted_class = CLASS_NAMES[idx],
        class_index     = idx,
        probabilities   = {n: float(p) for n, p in zip(CLASS_NAMES, probs)},
    )