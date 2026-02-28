from fastapi import FastAPI
from app.inference import predict
from app.schemas import PredictRequest, PredictResponse

app = FastAPI(title="CIFAR-10 Classifier")


@app.post("/predict", response_model=PredictResponse)
def predict_endpoint(payload: PredictRequest):
    # Skicka bilden genom modellen och få tillbaka klass + konfidens
    label, class_index, confidence = predict(payload.image_base64)
    return PredictResponse(
        prediction=label,
        class_index=class_index,
        confidence=confidence,
    )
