# captioner/llava_text.py

import requests
from event_logging.json_logger import log_llava_api_call
from config.config import MOOD_SNAPSHOT_FOLDER


def query_llava_text(prompt: str) -> str:
    """
    Sends a text-only prompt to LLaVA without any image.
    Returns the model's response as plain text.
    """
    payload = {"model": "llava", "prompt": prompt, "stream": False}

    try:
        resp = requests.post("http://localhost:11434/api/generate", json=payload, timeout=10)
        resp.raise_for_status()
        response_text = resp.json().get("response", "[No response]")
        
        # Log the successful API call
        log_llava_api_call(
            prompt=prompt,
            model="llava",
            image_path=None,
            response=response_text,
            success=True,
            timeout=10,
            log_dir=MOOD_SNAPSHOT_FOLDER
        )
        
        return response_text
    except Exception as e:
        error_msg = f"[⚠️ LLaVA text query failed: {e}]"
        
        # Log the failed API call
        log_llava_api_call(
            prompt=prompt,
            model="llava",
            image_path=None,
            response=None,
            success=False,
            error_message=str(e),
            timeout=10,
            log_dir=MOOD_SNAPSHOT_FOLDER
        )
        
        return error_msg
