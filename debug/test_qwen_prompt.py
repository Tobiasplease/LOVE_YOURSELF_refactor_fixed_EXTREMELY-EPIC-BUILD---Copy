#!/usr/bin/env python3

# Test what prompt is actually being generated for Qwen

from captioner.prompts import build_caption_prompt, build_qwen_scene_prompt
from config import config

print(f"Current model: {config.OLLAMA_MODEL}")

# Mock agent for testing
class MockAgent:
    def __init__(self):
        self.last_caption = "The room is dimly lit, with a warm yellow hue"
        self.model = type('MockModel', (), {'model_name': 'qwen2.5vl:7b'})()
    
    def temporal_prompt_lines(self):
        return ['day 0', 'awake 15m']

agent = MockAgent()
mood_vector = (0.5, 0.3, 0.7)  # valence, arousal, clarity

print("\n=== TESTING QWEN PROMPT DISPATCH ===")
prompt = build_caption_prompt(agent, 0.5, 0.2, 0.8, "The room is dimly lit")
print(prompt)

print("\n=== TESTING DIRECT QWEN SCENE PROMPT ===")
qwen_prompt = build_qwen_scene_prompt(agent, mood_vector, "The room is dimly lit")
print(qwen_prompt)

print("\n=== END TEST ===")
