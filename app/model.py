from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

_ARTIFACT_PATH = (
    Path(__file__).resolve().parent.parent / "artifacts" / "model.ts"
)

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

_model: Optional[torch.jit.RecursiveScriptModule] = None


class SimpleCNN(nn.Module):
    """Samma arkitektur som i Lab2-Ramverk (CIFAR-10)."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(32 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 10)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


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
    """Skapar SimpleCNN med slumpmässiga vikter och exporterar till TorchScript."""
    model = SimpleCNN()
    scripted = torch.jit.script(model)
    scripted.save(str(path))
    return scripted
