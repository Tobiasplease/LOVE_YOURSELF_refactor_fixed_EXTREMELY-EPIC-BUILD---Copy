# tone.py

from typing import List
from captioner.llava_text import query_llava_text

def infer_tone_from_text(text: str) -> List[str]:
    """
    Uses LLaVA to analyze tone from a caption.
    Returns 1–2 adjectives like ['dreamy', 'critical']
    """
    prompt = (
        "Analyze the tone of the following internal monologue. "
        "Return one or two adjectives describing its emotional or stylistic tone. "
        "Do not explain. Do not include extra words.\n\n"
        f"Sentence: \"{text.strip()}\""
    )

    raw = query_llava_text(prompt).strip().lower()
    adjectives = [w.strip() for w in raw.split(",") if w]
    return adjectives or ["neutral"]

def format_tone_descriptor(adjectives: List[str]) -> str:
    if not adjectives:
        return ""
    if adjectives == ["neutral"]:
        return "in a neutral tone"
    return "in a " + ", ".join(adjectives) + " tone"
