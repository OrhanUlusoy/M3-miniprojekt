from __future__ import annotations

import torch

from app.model import get_model


def predict(features: list[float]) -> float:
    model = get_model()

    x = torch.tensor([features], dtype=torch.float32)
    with torch.no_grad():
        y = model(x)

    return float(y.squeeze().item())
