#!/usr/bin/env python3
"""
Direct test of TinyLlama emotional voice generation
"""

from utils.ollama import query_ollama

def test_tinyllama_emotional_voice():
    """Test TinyLlama with chaos settings for emotional response."""
    
    # Test prompt
    prompt = "Scene: person desk laptop window\n\nGut feeling (3-5 words max):"
    
    # High chaos settings for authentic emotional response
    chaos_options = {
        "temperature": 1.4,
        "top_p": 0.7,
        "repeat_penalty": 1.8,
        "num_ctx": 512,  # Small context
        "stop": ["\n", ".", "!", "?", ","]  # Stop early
    }
    
    print(f"Testing TinyLlama emotional voice generation...")
    print(f"Prompt: {prompt}")
    print(f"Options: {chaos_options}")
    
    try:
        response = query_ollama(
            prompt=prompt,
            model="tinyllama:latest",
            timeout=10,
            options=chaos_options,
            system_prompt="You respond with immediate gut feelings. No analysis, just visceral reaction."
        )
        
        print(f"Raw response: '{response}'")
        
        if response:
            clean_response = response.strip()[:30]
            print(f"Clean response: '{clean_response}'")
            return clean_response
        else:
            print("Empty response!")
            return ""
            
    except Exception as e:
        print(f"Error: {e}")
        return ""

if __name__ == "__main__":
    for i in range(3):
        print(f"\n--- Test {i+1} ---")
        result = test_tinyllama_emotional_voice()
        print(f"Result: '{result}'")
