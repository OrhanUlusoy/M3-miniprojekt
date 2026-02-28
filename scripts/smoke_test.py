"""Smoke-test: startar API:t, skickar en testbild till POST /predict, verifierar svaret."""

from __future__ import annotations

import base64
import io
import subprocess
import sys
import time
import urllib.request
import urllib.error
import json

from PIL import Image


def _make_test_image_base64() -> str:
    """Skapar en liten 32x32 röd bild som base64."""
    img = Image.new("RGB", (32, 32), color=(255, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def main() -> None:
    # Starta uvicorn i bakgrunden
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", "8123"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    try:
        url = "http://127.0.0.1:8123/predict"
        image_b64 = _make_test_image_base64()
        payload = json.dumps({"image_base64": image_b64}).encode()

        for attempt in range(20):
            time.sleep(1)
            try:
                req = urllib.request.Request(
                    url,
                    data=payload,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=10) as resp:
                    body = json.loads(resp.read())
                    print(f"OK — POST /predict → {body}")
                    assert "prediction" in body, "Saknar 'prediction' i svaret"
                    assert "class_index" in body, "Saknar 'class_index' i svaret"
                    assert "confidence" in body, "Saknar 'confidence' i svaret"
                    assert isinstance(body["prediction"], str), f"Oväntat typ: {type(body['prediction'])}"
                    assert 0 <= body["class_index"] <= 9, f"Ogiltigt class_index: {body['class_index']}"
                    print("PASSED ✓")
                    return
            except (urllib.error.URLError, ConnectionRefusedError, OSError):
                if attempt < 19:
                    continue
                raise
        print("FAILED — servern svarade aldrig")
        sys.exit(1)
    finally:
        proc.terminate()
        proc.wait(timeout=5)


if __name__ == "__main__":
    main()
