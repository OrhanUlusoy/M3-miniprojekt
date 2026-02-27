# M3 miniprojekt – FastAPI + PyTorch (TorchScript/ONNX) + Docker

## Mål
- Exponera en PyTorch-modell via ett FastAPI-API.
- Exportera modellen till TorchScript (eller ONNX).
- Köra allt i Docker, med beroenden installerade via **uv**.

## API
- `POST /predict`
  - Input (JSON):
    ```json
    {"features": [1.0, 2.0, 3.0]}
    ```
  - Output (JSON):
    ```json
    {"prediction": 1.4}
    ```

## Köra lokalt
1. Installera uv: https://docs.astral.sh/uv/
2. Installera deps:
   - `uv pip install -r requirements.txt`
3. Starta API:
   - `uvicorn app.main:app --reload`
4. Testa:
   - `curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d "{\"features\": [1,2,3]}"`

## Bygga och köra container
- Bygg: `docker build -t m3-miniprojekt .`
- Kör: `docker run --rm -p 8000:8000 m3-miniprojekt`

## Modell (att byta till er K2-modell)
- Demo-export finns i [scripts/export_torchscript.py](scripts/export_torchscript.py).
- Docker-bygget kör exporten och skapar `artifacts/model.ts`.
- Byt ut demo-modellen mot er egen och se till att export-scriptet genererar TorchScript/ONNX från er tränade modell.

## Pull Requests (länka minst två med kodgranskning)
- PR 1: <länk>
- PR 2: <länk>
