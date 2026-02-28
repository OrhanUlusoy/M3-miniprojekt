from __future__ import annotations

"""Exporterar SimpleCNN (CIFAR-10) till TorchScript.

Kör: python scripts/export_torchscript.py
Output: artifacts/model.ts
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# Lägg till Lab2-Ramverk i path så vi kan importera modellen
LAB2_DIR = Path(__file__).resolve().parent.parent.parent / "Lab2-Ramverk"


class SimpleCNN(nn.Module):
    """Samma arkitektur som i Lab2-Ramverk."""

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


def main() -> None:
    artifact_path = Path(__file__).resolve().parent.parent / "artifacts" / "model.ts"
    artifact_path.parent.mkdir(parents=True, exist_ok=True)

    model = SimpleCNN()

    # Försök ladda tränade vikter om de finns
    weights_path = LAB2_DIR / "model.pth"
    if weights_path.exists():
        state = torch.load(str(weights_path), map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        print(f"Laddade vikter från {weights_path}")
    else:
        print(f"OBS: Hittade inga tränade vikter på {weights_path} — exporterar med slumpmässiga vikter.")

    model.eval()
    scripted = torch.jit.script(model)
    scripted.save(str(artifact_path))
    print(f"Wrote: {artifact_path}")


if __name__ == "__main__":
    main()
