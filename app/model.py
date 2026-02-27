from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
from torch import nn

_ARTIFACT_PATH = (
    Path(__file__).resolve().parent.parent / "artifacts" / "model.ts"
)

_model: Optional[torch.jit.RecursiveScriptModule] = None


def get_model() -> torch.jit.RecursiveScriptModule:
    global _model
    if _model is not None:
        return _model

    if not _ARTIFACT_PATH.exists():
        _ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
        _model = _create_and_export_default_model(_ARTIFACT_PATH)
    else:
        _model = torch.jit.load(str(_ARTIFACT_PATH), map_location="cpu")

    _model.eval()
    return _model


def _create_and_export_default_model(path: Path) -> torch.jit.RecursiveScriptModule:
    """Skapar en minimal demo-modell och exporterar den som TorchScript.

    Byt ut detta mot er riktiga K2-modell + exportsteg.
    """

    class ToyRegressor(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = nn.Linear(3, 1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.linear(x)

    model = ToyRegressor()

    with torch.no_grad():
        model.linear.weight.copy_(torch.tensor([[0.1, 0.2, 0.3]], dtype=torch.float32))
        model.linear.bias.copy_(torch.tensor([0.0], dtype=torch.float32))

    scripted = torch.jit.script(model)
    scripted.save(str(path))
    return scripted
