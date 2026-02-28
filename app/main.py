from __future__ import annotations

from fastapi import FastAPI

from app.inference import predict
from app.schemas import PredictRequest, PredictResponse

app = FastAPI(title="M3 Miniprojekt – CIFAR-10 Classifier")


@app.post("/predict", response_model=PredictResponse)
def predict_endpoint(payload: PredictRequest) -> PredictResponse:
    label, class_index, confidence = predict(payload.image_base64)
    return PredictResponse(prediction=label, class_index=class_index, confidence=confidence)
