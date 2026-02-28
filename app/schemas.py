from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    image_base64: str = Field(
        ..., description="Base64-kodad bild (PNG/JPEG). Skalas till 32x32 av API:t."
    )


class PredictResponse(BaseModel):
    prediction: str = Field(..., description="Förutsagd CIFAR-10-klass, t.ex. 'cat'.")
    class_index: int = Field(..., description="Klassindex (0-9).")
    confidence: float = Field(..., description="Softmax-konfidens för den förutsagda klassen.")
