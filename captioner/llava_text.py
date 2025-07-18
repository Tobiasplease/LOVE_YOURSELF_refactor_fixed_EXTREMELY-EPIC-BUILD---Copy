# captioner/llava_text.py

from ollama import query_ollama
from config.config import MOOD_SNAPSHOT_FOLDER


def query_llava_text(prompt: str) -> str:
    """
    Sends a text-only prompt to LLaVA without any image.
    Returns the model's response as plain text.
    """
    return query_ollama(
        prompt=prompt,
        model="llava",
        image=None,
        timeout=10,
        log_dir=MOOD_SNAPSHOT_FOLDER
    )
