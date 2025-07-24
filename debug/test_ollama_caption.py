import time
from ollama import query_ollama

# === Path to any test image ===
image_path = "/Users/jbe/Desktop/505376203_697336169816350_2001727864992071742_n.jpg"

print("Sending request to ollama...")

try:
    start = time.time()
    response = query_ollama(prompt="What do you see?", model="mistral", image=image_path, timeout=20, log_dir="mood_snapshots")
    duration = time.time() - start
    print(f"✅ ollama responded in {duration:.2f} seconds.")
    print("Response:", response)

except Exception as e:
    print("⚠️ Exception:", e)
