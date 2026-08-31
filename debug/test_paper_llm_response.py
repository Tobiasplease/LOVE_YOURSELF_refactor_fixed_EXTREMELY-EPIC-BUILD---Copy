#!/usr/bin/env python3
import base64
import sys

import requests

image_path = sys.argv[1] if len(sys.argv) > 1 else "event_log/df99f284-images/paper_check_1759511573.jpg"

prompt = """Please examine this image carefully using these steps:

1. First, scan the entire image for any visible text, words, or writing
2. Look specifically for the text "EMPTY" written anywhere in the image
3. Answer YES if you can see "EMPTY" text, NO if you cannot see it

Answer: YES or NO"""

with open(image_path, "rb") as f:
    img_b64 = base64.b64encode(f.read()).decode("utf-8")

payload = {
    "model": "llava:7b-v1.6-mistral-q5_1",
    "prompt": prompt,
    "stream": False,
    "images": [img_b64],
    "options": {
        "temperature": 0.1,
        "top_p": 0.9,
        "top_k": 10,
        "repeat_penalty": 1.0,
    },
}

response_obj = requests.post("http://localhost:11434/api/generate", json=payload, timeout=60)
response = response_obj.json().get("response", "")

print(f"Image: {image_path}")
print(f"Response: {response}")
print(f"Response upper: {response.strip().upper()}")
print(f"Contains YES: {'YES' in response.strip().upper()}")
print(f"Contains NO: {'NO' in response.strip().upper()}")
