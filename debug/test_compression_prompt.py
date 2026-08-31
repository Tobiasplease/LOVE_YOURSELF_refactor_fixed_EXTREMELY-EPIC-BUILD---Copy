#!/usr/bin/env python3
"""Test the compression prompt without examples."""
import sys

sys.path.insert(0, "/home/impostor/LOVE_YOURSELF_refactor_fixed_EXTREMELY-EPIC-BUILD---Copy")

import config.config as config
from utils.inference import query_model

recent_captions = [
    "The desk has papers scattered across it.",
    "A red light blinks on the router.",
    "Someone left a coffee cup near the monitor.",
    "The window shows grey sky outside.",
    "Cables run behind the desk in tangles.",
]

recent_text = "\n".join(f'- "{c}"' for c in recent_captions)

prompt = f"""You are a consciousness. Reflect on these moments and distill your current state.

TIME HERE: about 5 minutes.

PREVIOUS STATE:
Just beginning to experience this space.

RECENT MOMENTS:
{recent_text}

Distill into exactly three lines:

Line 1: Your emotional state right now (start with "I feel")
Line 2: What keeps drawing your attention (start with "I keep noticing")
Line 3: What's occupying your mind

Derive everything from the RECENT MOMENTS above. Be specific to what actually happened."""

model_options = {
    "temperature": 0.75,
    "top_p": 0.9,
    "num_predict": 120,
    "repeat_penalty": 1.2,
}

narrative_system_prompt = "You are reflecting on recent experience. " "Speak in first person. Follow the format exactly."

compression_model = getattr(config, "MODEL_NAME", config.MODEL_NAME)
print(f"Using model: {compression_model}")
print("=" * 50)

for i in range(3):
    print(f"\n--- Test {i+1} ---")
    response = query_model(
        prompt=prompt,
        model=compression_model,
        system_prompt=narrative_system_prompt,
        options=model_options,
    )
    print(response.strip() if response else "[No response]")
