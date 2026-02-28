from fastapi import FastAPI, HTTPException
from app.inference import predict
from app.schemas import PredictRequest, PredictResponse

app = FastAPI(title="CIFAR-10 Classifier")


@app.post("/predict", response_model=PredictResponse)
def predict_endpoint(payload: PredictRequest):
    # Kör predict och fånga valideringsfel så vi ger ett tydligt svar istället för 500
    try:
        label, class_index, confidence = predict(payload.image_base64)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return PredictResponse(
        prediction=label,
        class_index=class_index,
        confidence=confidence,
    )
