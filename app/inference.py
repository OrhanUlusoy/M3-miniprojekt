from __future__ import annotations

import base64
import io

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

from app.model import get_model, CIFAR10_CLASSES

_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
])


def predict(image_base64: str) -> tuple[str, int, float]:
    """Tar en base64-kodad bild, returnerar (klass-namn, klass-index, konfidens)."""
    image_bytes = base64.b64decode(image_base64)
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    tensor = _transform(image).unsqueeze(0)  # (1, 3, 32, 32)

    model = get_model()
    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1)
        confidence, class_idx = torch.max(probs, dim=1)

    idx = int(class_idx.item())
    return CIFAR10_CLASSES[idx], idx, float(confidence.item())
