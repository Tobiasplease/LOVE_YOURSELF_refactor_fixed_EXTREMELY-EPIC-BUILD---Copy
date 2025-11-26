#!/usr/bin/env python3
"""
Test script to check if ollama supports multiple images for comparison.
"""

import base64
import json
import requests
import os
from typing import List

def query_ollama_multi_image(prompt: str, image_paths: List[str], model: str = "llava:7b-v1.6-mistral-q5_1") -> str:
    """Test function to send multiple images to ollama."""

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "images": []
    }

    # Add all images to payload
    for image_path in image_paths:
        if os.path.exists(image_path):
            with open(image_path, "rb") as img_file:
                img_bytes = img_file.read()
                img_b64 = base64.b64encode(img_bytes).decode("utf-8")
                payload["images"].append(img_b64)
            print(f"Added image: {image_path}")
        else:
            print(f"Image not found: {image_path}")

    print(f"Sending {len(payload['images'])} images to ollama...")

    try:
        response = requests.post("http://localhost:11434/api/generate", json=payload, timeout=30)
        response.raise_for_status()
        result = response.json().get("response", "")
        return result
    except Exception as e:
        return f"Error: {e}"

def test_paper_detection_comparison():
    """Test paper detection with reference images."""

    paper_present_path = "./calibration/paper_present.jpg"
    paper_absent_path = "./calibration/paper_absent.jpg"

    if not os.path.exists(paper_present_path) or not os.path.exists(paper_absent_path):
        print("❌ Reference images not found!")
        return False

    print("🔍 Testing multi-image paper detection...")

    # Test 1: Can it see multiple images?
    prompt1 = """You are viewing multiple images. How many images can you see? Describe each one briefly."""

    response1 = query_ollama_multi_image(prompt1, [paper_present_path, paper_absent_path])
    print(f"\n📊 Multi-image test response:\n{response1}\n")

    # Test 2: Paper detection comparison
    prompt2 = """Compare these two images of the same drawing area:

IMAGE 1: Shows the drawing area WITH paper properly positioned
IMAGE 2: Shows the drawing area WITHOUT paper (bare surface)

Now I will show you a current image, and you need to determine which reference image it most closely resembles.

Looking at the current scene, does it look more like:
A) Image 1 (paper present) - white paper visible on the surface
B) Image 2 (paper absent) - bare orange/yellow surface visible

Respond with:
PAPER: YES/NO
CONFIDENCE: 0.0-1.0
REASON: Which reference image it matches and why

Be specific about surface texture, color, and what you can see."""

    # For now, test with one of the reference images as "current"
    response2 = query_ollama_multi_image(prompt2, [paper_present_path, paper_absent_path, paper_present_path])
    print(f"\n📄 Paper detection test response:\n{response2}\n")

    return True

if __name__ == "__main__":
    test_paper_detection_comparison()