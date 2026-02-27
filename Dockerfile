FROM python:3.11-slim

WORKDIR /app

# Install uv (dependency manager)
RUN pip install --no-cache-dir uv

# Install dependencies
COPY requirements.txt ./requirements.txt
RUN uv pip install --system --no-cache -r requirements.txt

# Copy app code
COPY . .

# Export TorchScript model (demo) during build so the container works out-of-the-box
RUN python scripts/export_torchscript.py

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
