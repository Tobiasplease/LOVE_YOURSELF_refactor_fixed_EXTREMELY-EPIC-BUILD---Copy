"""Show exactly what the stream drawing pipeline's intent call would see,
without running the model. Monkeypatches query_model to capture the prompt,
uses the real on-disk memory state (context compressor, drawing memory,
reflections in ChromaDB).

    python debug/test_drawing_intent_prompt.py [image_path]
"""

import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import utils.inference as inference

captured = []


def fake_query_model(prompt, image=None, system_prompt=None, prompt_type=None, **kwargs):
    captured.append({"type": prompt_type, "system": system_prompt, "prompt": prompt, "image": image})
    if prompt_type == "drawing_intent":
        return "I draw the person hunched at the screen, one long line for the spine."
    return "Black ink line drawing on white paper. A hunched figure at a desk."


inference.query_model = fake_query_model

from captioner.prompts import stream_drawing_analysis

memory_ref = SimpleNamespace(
    _stream=[
        "The figure at the computer leans closer to the screen, shoulders drawn up.",
        "The red foam finger has not moved; the light on it has.",
        "I keep circling the same hesitation instead of pressing the pen down.",
    ]
)

image_path = sys.argv[1] if len(sys.argv) > 1 else None
stream_drawing_analysis(memory_ref, extra=None, image_path=image_path)

for call in captured:
    print("=" * 70)
    print(f"CALL: {call['type']}   image attached: {bool(call['image'])}")
    print("-" * 70)
    print("SYSTEM:", call["system"])
    print("-" * 70)
    print(call["prompt"])
