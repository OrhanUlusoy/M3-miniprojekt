from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    features: list[float] = Field(
        ..., description="Numeriska features i samma ordning som modellen förväntar sig."
    )


class PredictResponse(BaseModel):
    prediction: float
