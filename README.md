# M3 miniprojekt – CIFAR-10 klassificering

Bildklassificering med en CNN tränad på CIFAR-10. Modellen exporteras till TorchScript och servas via FastAPI i en Docker-container.

## Modellen

SimpleCNN från min Lab2 (Ramverk-kursen). Tar in en 32x32 RGB-bild och klassificerar den som en av:

`airplane | automobile | bird | cat | deer | dog | frog | horse | ship | truck`

## API

**POST /predict**

Skicka en base64-kodad bild och få tillbaka vilken klass modellen tror det är:

```json
// Request
{"image_base64": "iVBORw0KGgo..."}

// Response
{"prediction": "cat", "class_index": 3, "confidence": 0.87}
```

## Kom igång

```bash
# Installera dependencies
uv sync

# Exportera modellen till TorchScript
python scripts/export_torchscript.py

# Starta servern
uvicorn app.main:app --reload
```

## Docker

```bash
docker build -t m3-miniprojekt .
docker run --rm -p 8000:8000 m3-miniprojekt
```

## Testa att det funkar

```bash
python scripts/smoke_test.py
```

## Projektstruktur

```
app/
  main.py          # FastAPI-app med /predict-endpoint
  inference.py     # Bildförbehandling + inferens
  model.py         # Modelldefinition (SimpleCNN) + laddning av TorchScript
  schemas.py       # Request/response-modeller (Pydantic)
scripts/
  export_torchscript.py   # Exporterar modellen till TorchScript
  smoke_test.py           # Snabbtest som verifierar att API:t svarar
artifacts/
  model.ts         # Exporterad TorchScript-modell (genereras, versionshanteras ej)
```

## Reflektioner

Modellen är en ganska enkel CNN med bara två conv-lager, så träffsäkerheten på CIFAR-10 hamnar runt 60-65%. Det räcker för att visa att pipeline:n fungerar, men i ett riktigt projekt hade man velat använda en djupare arkitektur (t.ex. ResNet) eller åtminstone data augmentation.

TorchScript valdes framför ONNX mest för att det var enklast — modellen kunde scripta:s direkt utan problem. ONNX hade varit bättre om man ville köra inferens utan PyTorch (t.ex. med ONNX Runtime), men det behövdes inte här.

En sak jag hade gjort annorlunda är att spara vikterna i Lab2 redan från början, nu fick jag exportera med slumpmässiga vikter eftersom tränade vikter inte sparades till disk i det projektet.

## Pull Requests
- PR 1: https://github.com/OrhanUlusoy/M3-miniprojekt/pull/1
- PR 2: https://github.com/OrhanUlusoy/M3-miniprojekt/pull/2
