#!/usr/bin/env python3
"""
Test the simplified consciousness system with actual Ollama model.
"""

import time
import requests
import json
from captioner.prompts import build_simple_caption_prompt


class MockMemory:
    """Mock memory for testing."""
    def get_top_motifs(self, k):
        return ["ceiling_damage", "light_fixtures", "desk_activity"]


class MockAgent:
    """Mock agent for testing."""
    
    def __init__(self):
        self.true_session_start = time.time() - 1800  # 30 minutes ago
        self.emotional_journey = ["calm", "alert", "engaged", "curious"]
        self.memory_ref = MockMemory()


def call_ollama_with_image(prompt: str, image_path: str = None) -> str:
    """Call Ollama with the prompt and optional image."""
    
    url = "http://localhost:11434/api/generate"
    
    # Mock image data for testing (empty for text-only)
    data = {
        "model": "llava:7b-v1.6-mistral-q5_1",
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.7,
            "top_p": 0.9
        }
    }
    
    try:
        response = requests.post(url, json=data, timeout=30)
        response.raise_for_status()
        
        result = response.json()
        return result.get('response', 'No response')
        
    except Exception as e:
        return f"Error: {e}"


def test_with_ollama():
    """Test simplified consciousness with Ollama."""
    
    print("=== Testing Simplified Consciousness with Ollama ===\n")
    
    # Create mock agent
    agent = MockAgent()
    
    # Test case: curious state
    mood_vector = (0.3, 0.6, 0.8)  # curious and alert
    last_caption = "The room appears calm and organized."
    
    # Generate prompt
    prompt = build_simple_caption_prompt(agent, mood_vector, last_caption)
    
    print("Generated Prompt:")
    print(prompt)
    print("\n" + "="*60 + "\n")
    
    print("Calling Ollama...")
    response = call_ollama_with_image(prompt)
    
    print("Ollama Response:")
    print(response)
    print("\n" + "="*60 + "\n")
    
    # Test another mood
    print("=== Testing Different Mood ===\n")
    
    mood_vector = (0.8, 0.7, 0.9)  # energized and engaged
    last_caption = response[:100] + "..."  # Use part of previous response
    
    prompt = build_simple_caption_prompt(agent, mood_vector, last_caption)
    
    print("New Prompt:")
    print(prompt)
    print("\n" + "="*40 + "\n")
    
    response2 = call_ollama_with_image(prompt)
    
    print("Second Response:")
    print(response2)


if __name__ == "__main__":
    test_with_ollama()
