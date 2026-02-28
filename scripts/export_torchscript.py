# Script för att exportera CIFAR-10-modellen till TorchScript.
# Kör med: python scripts/export_torchscript.py
# Sparar modellen som artifacts/model.ts

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# Sökväg till Lab2-repot där vikterna eventuellt finns sparade
LAB2_DIR = Path(__file__).resolve().parent.parent.parent / "Lab2-Ramverk"


class SimpleCNN(nn.Module):
    # Samma CNN som i Lab2 — två conv-lager + två FC-lager

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(32 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 10)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def main():
    artifact_path = Path(__file__).resolve().parent.parent / "artifacts" / "model.ts"
    artifact_path.parent.mkdir(parents=True, exist_ok=True)

    model = SimpleCNN()

    # Kolla om det finns sparade vikter från Lab2-träningen
    weights_path = LAB2_DIR / "model.pth"
    if weights_path.exists():
        state = torch.load(str(weights_path), map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        print(f"Laddade vikter från {weights_path}")
    else:
        # Inga vikter — använder slumpmässiga (fungerar ändå för att testa API:t)
        print(f"Hittade inga vikter på {weights_path}, kör med random weights")

    # Sätt modellen i eval-läge och exportera till TorchScript
    model.eval()
    scripted = torch.jit.script(model)
    scripted.save(str(artifact_path))
    print(f"Sparade modell till {artifact_path}")


if __name__ == "__main__":
    main()
