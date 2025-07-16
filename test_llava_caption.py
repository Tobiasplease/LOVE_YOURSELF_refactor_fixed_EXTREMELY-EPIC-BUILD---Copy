import requests
import base64
import json
import time

# === Path to any test image ===
image_path = r"C:\Users\tobia\Downloads\lint_caption_display_window_fully_synced\lint_caption_display_window_fixed_final\mood_snapshots\mood_1752414950.jpg"

# === Read and encode image ===
with open(image_path, "rb") as img_file:
    img_bytes = img_file.read()
    img_b64 = base64.b64encode(img_bytes).decode("utf-8")

# === Compose request ===
payload = {
    "model": "llava",
    "prompt": "What do you see?",
    "images": [img_b64],
    "stream": False
}

print("Sending request to LLaVA...")

try:
    start = time.time()
    res = requests.post(
        "http://localhost:11434/api/generate",
        json=payload,
        timeout=20  # ⏱ Increase this if needed
    )
    duration = time.time() - start
    print(f"✅ LLaVA responded in {duration:.2f} seconds.")

    if res.ok:
        print("Response:", res.json())
    else:
        print("❌ Error:", res.status_code, res.text)

except Exception as e:
    print("⚠️ Exception:", e)
