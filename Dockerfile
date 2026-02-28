FROM python:3.11-slim

WORKDIR /app

# Installera uv för att hantera dependencies
RUN pip install --no-cache-dir uv

# Kopiera in requirements och installera allt
# Använder requirements-docker.txt med CPU-only torch (mycket mindre image)
COPY requirements-docker.txt ./requirements-docker.txt
RUN uv pip install --system --no-cache -r requirements-docker.txt

# Kopiera resten av projektet
COPY . .

# Exportera modellen till TorchScript redan vid bygget
RUN python scripts/export_torchscript.py

EXPOSE 8000

# Starta API-servern
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
