# captioner/llava_text.py

import requests

def query_llava_text(prompt: str) -> str:
    """
    Sends a text-only prompt to LLaVA without any image.
    Returns the model's response as plain text.
    """
    payload = {
        "model": "llava",
        "prompt": prompt,
        "stream": False
    }

    try:
        resp = requests.post("http://localhost:11434/api/generate", json=payload, timeout=10)
        resp.raise_for_status()
        return resp.json().get("response", "[No response]")
    except Exception as e:
        return f"[⚠️ LLaVA text query failed: {e}]"
