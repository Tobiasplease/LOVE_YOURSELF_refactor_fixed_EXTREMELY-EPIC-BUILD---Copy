import requests
import base64

# import json
import time

# === Path to any test image ===
image_path = "/Users/jbe/Desktop/505376203_697336169816350_2001727864992071742_n.jpg"

# === Read and encode image ===
with open(image_path, "rb") as img_file:
    img_bytes = img_file.read()
    img_b64 = base64.b64encode(img_bytes).decode("utf-8")

# === Compose request ===
payload = {"model": "llava", "prompt": "What do you see?", "images": [img_b64], "stream": False}

print("Sending request to LLaVA...")

try:
    start = time.time()
    res = requests.post("http://localhost:11434/api/generate", json=payload, timeout=20)  # ⏱ Increase this if needed
    duration = time.time() - start
    print(f"✅ LLaVA responded in {duration:.2f} seconds.")

    if res.ok:
        print("Response:", res.json())
    else:
        print("❌ Error:", res.status_code, res.text)

except Exception as e:
    print("⚠️ Exception:", e)
