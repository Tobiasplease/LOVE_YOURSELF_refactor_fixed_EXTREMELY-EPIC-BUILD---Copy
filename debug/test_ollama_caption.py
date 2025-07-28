import time
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.ollama import query_ollama

# === Path to any test image ===
image_path = "/Users/jbe/Desktop/505376203_697336169816350_2001727864992071742_n.jpg"

# Check if image exists, if not try to find any image in mood_snapshots
if not os.path.exists(image_path):
    mood_folder = os.path.join(os.path.dirname(os.path.dirname(__file__)), "mood_snapshots")
    if os.path.exists(mood_folder):
        image_files = [f for f in os.listdir(mood_folder) if f.endswith(('.jpg', '.jpeg', '.png'))]
        if image_files:
            image_path = os.path.join(mood_folder, image_files[0])
            print(f"Using test image: {image_path}")
        else:
            print("No test images found in mood_snapshots folder")
            image_path = None
    else:
        print("No mood_snapshots folder found")
        image_path = None

if image_path and os.path.exists(image_path):
    print("Testing streaming response...")

    try:
        start = time.time()
        response = query_ollama(prompt="What do you see in this image? Describe it in detail.", model="mistral", image=image_path, timeout=20, log_dir="mood_snapshots", stream=True)
        duration = time.time() - start
        print(f"✅ ollama responded in {duration:.2f} seconds with streaming.")
        print("Response:", response)

    except Exception as e:
        print("⚠️ Exception:", e)

    print("\nTesting non-streaming response...")

    try:
        start = time.time()
        response = query_ollama(prompt="What do you see in this image? Describe it in detail.", model="mistral", image=image_path, timeout=20, log_dir="mood_snapshots", stream=False)
        duration = time.time() - start
        print(f"✅ ollama responded in {duration:.2f} seconds without streaming.")
        print("Response:", response)

    except Exception as e:
        print("⚠️ Exception:", e)
        
else:
    print("❌ No test image available - cannot test vision capabilities")
    
    # Test text-only processing
    print("\nTesting text-only processing with streaming...")
    try:
        start = time.time()
        response = query_ollama(prompt="Describe a peaceful morning scene.", model="mistral", timeout=20, log_dir="mood_snapshots", stream=True)
        duration = time.time() - start
        print(f"✅ ollama responded in {duration:.2f} seconds with streaming.")
        print("Response:", response)
    except Exception as e:
        print("⚠️ Exception:", e)
