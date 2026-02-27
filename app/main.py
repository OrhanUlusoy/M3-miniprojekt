from __future__ import annotations

from fastapi import FastAPI

from app.inference import predict
from app.schemas import PredictRequest, PredictResponse

app = FastAPI(title="M3 Miniprojekt")


@app.post("/predict", response_model=PredictResponse)
def predict_endpoint(payload: PredictRequest) -> PredictResponse:
    prediction = predict(payload.features)
    return PredictResponse(prediction=prediction)
