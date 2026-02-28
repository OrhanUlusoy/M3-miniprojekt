import base64
import io

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

from app.model import get_model, CIFAR10_CLASSES

# Förbehandling: skala bilden till 32x32 (CIFAR-10-storlek) och konvertera till tensor
preprocess = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),  # skalar pixelvärden till [0, 1] och gör om till CHW-format
])


def predict(image_base64):
    # Avkoda base64-strängen till en PIL-bild
    raw_bytes = base64.b64decode(image_base64)
    image = Image.open(io.BytesIO(raw_bytes)).convert("RGB")

    # Gör om bilden till en tensor med rätt shape för modellen
    tensor = preprocess(image).unsqueeze(0)  # lägg till batch-dimension -> (1, 3, 32, 32)

    model = get_model()

    # Kör inferens utan att spåra gradienter (snabbare + sparar minne)
    with torch.no_grad():
        logits = model(tensor)

        # Softmax ger oss sannolikheter för varje klass
        probs = F.softmax(logits, dim=1)

        # Plocka ut klassen med högst sannolikhet
        confidence, class_idx = torch.max(probs, dim=1)

    idx = int(class_idx.item())
    return CIFAR10_CLASSES[idx], idx, float(confidence.item())
