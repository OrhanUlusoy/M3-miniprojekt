from pydantic import BaseModel


# Request-body för /predict
class PredictRequest(BaseModel):
    image_base64: str  # base64-kodad PNG eller JPEG


# Svar från /predict
class PredictResponse(BaseModel):
    prediction: str     # t.ex. "cat", "ship" osv
    class_index: int    # 0-9 (index i CIFAR-10)
    confidence: float   # sannolikhet (softmax) för den valda klassen
