# M3 miniprojekt – CIFAR-10 Classifier (FastAPI + TorchScript + Docker)

## Mål
- Exponera en SimpleCNN (CIFAR-10) via ett FastAPI-API.
- Exportera modellen till TorchScript.
- Köra allt i Docker, med beroenden installerade via **uv**.

## API
- `POST /predict`
  - Input (JSON):
    ```json
    {"image_base64": "<base64-kodad PNG/JPEG-bild>"}
    ```
  - Output (JSON):
    ```json
    {"prediction": "cat", "class_index": 3, "confidence": 0.87}
    ```
  - Klasser: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck

## Köra lokalt
1. Installera uv: https://docs.astral.sh/uv/
2. Installera deps:
   - `uv sync`
3. Exportera modell:
   - `python scripts/export_torchscript.py`
4. Starta API:
   - `uvicorn app.main:app --reload`
5. Testa (skicka en base64-kodad bild):
   ```bash
   python -c "
   import base64, json, urllib.request
   from PIL import Image
   import io
   img = Image.new('RGB', (32,32), (255,0,0))
   buf = io.BytesIO(); img.save(buf, format='PNG')
   b64 = base64.b64encode(buf.getvalue()).decode()
   req = urllib.request.Request('http://127.0.0.1:8000/predict',
       data=json.dumps({'image_base64': b64}).encode(),
       headers={'Content-Type': 'application/json'}, method='POST')
   print(json.loads(urllib.request.urlopen(req).read()))
   "
   ```

## Bygga och köra container
- Bygg: `docker build -t m3-miniprojekt .`
- Kör: `docker run --rm -p 8000:8000 m3-miniprojekt`

## Smoke-test
```bash
python scripts/smoke_test.py
```

## Pull Requests (länka minst två med kodgranskning)
- PR 1: <länk>
- PR 2: <länk>
