# Snabbtest som kollar att API:t faktiskt funkar.
# Startar servern, skickar en testbild, och verifierar att svaret ser rimligt ut.

import base64
import io
import subprocess
import sys
import time
import urllib.request
import urllib.error
import json

from PIL import Image


def make_test_image():
    # Skapa en enkel röd 32x32-bild och koda den som base64
    img = Image.new("RGB", (32, 32), color=(255, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def main():
    # Starta API-servern på port 8123 så den inte krockar med nåt annat
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", "8123"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    try:
        url = "http://127.0.0.1:8123/predict"
        image_b64 = make_test_image()
        payload = json.dumps({"image_base64": image_b64}).encode()

        # Prova ansluta i upp till 20 sekunder (servern kan ta ett tag att starta)
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
                    print(f"Svar: {body}")

                    # Kolla att alla fält finns med
                    assert "prediction" in body
                    assert "class_index" in body
                    assert "confidence" in body

                    # Prediction ska vara en sträng (klassnamn)
                    assert isinstance(body["prediction"], str)

                    # Class index ska vara 0-9
                    assert 0 <= body["class_index"] <= 9

                    print("Allt OK!")
                    return
            except (urllib.error.URLError, ConnectionRefusedError, OSError):
                # Servern har inte startat ännu, vänta lite till
                if attempt < 19:
                    continue
                raise

        print("Servern svarade aldrig")
        sys.exit(1)
    finally:
        # Stäng ner servern
        proc.terminate()
        proc.wait(timeout=5)


if __name__ == "__main__":
    main()
