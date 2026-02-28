from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# Sökväg till den exporterade TorchScript-filen
ARTIFACT_PATH = Path(__file__).resolve().parent.parent / "artifacts" / "model.ts"

# De 10 klasserna i CIFAR-10 (samma ordning som i datasetet)
CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

# Cachad modell så vi inte laddar om den varje request
_cached_model = None


class SimpleCNN(nn.Module):
    # CNN för CIFAR-10, kopierad från min Lab2.
    # Två conv-lager med max-pooling, sen två fully connected.
    # Input: (batch, 3, 32, 32) -> Output: (batch, 10)

    def __init__(self):
        super().__init__()

        # Första conv-lagret: 3 färgkanaler in, 16 filter ut
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        # Andra conv-lagret: 16 in, 32 ut
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)

        # Efter 2x pooling (32->16->8) har vi 32 kanaler á 8x8 = 2048 features
        self.fc1 = nn.Linear(32 * 8 * 8, 128)
        # Sista lagret: 128 -> 10 klasser
        self.fc2 = nn.Linear(128, 10)

        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        # Conv + ReLU + Pooling (halverar spatiala dimensioner varje gång)
        x = self.pool(F.relu(self.conv1(x)))   # -> (batch, 16, 16, 16)
        x = self.pool(F.relu(self.conv2(x)))   # -> (batch, 32, 8, 8)

        # Platta ut till en vektor per bild
        x = torch.flatten(x, 1)

        # Fully connected-lager
        x = F.relu(self.fc1(x))
        x = self.fc2(x)  # raw logits, ingen softmax här
        return x


def get_model():
    # Returnerar den laddade TorchScript-modellen.
    # Om artefakten inte finns skapas en ny med slumpmässiga vikter
    # (bra för att testa att allt fungerar utan att behöva träna först).
    global _cached_model

    if _cached_model is not None:
        return _cached_model

    if not ARTIFACT_PATH.exists():
        # Ingen exporterad modell hittades, skapa en ny
        ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
        _cached_model = _export_fresh_model(ARTIFACT_PATH)
    else:
        # Ladda befintlig TorchScript-fil
        _cached_model = torch.jit.load(str(ARTIFACT_PATH), map_location="cpu")

    _cached_model.eval()
    return _cached_model


def _export_fresh_model(path):
    # Skapar en ny SimpleCNN och sparar som TorchScript
    model = SimpleCNN()
    scripted = torch.jit.script(model)
    scripted.save(str(path))
    return scripted
