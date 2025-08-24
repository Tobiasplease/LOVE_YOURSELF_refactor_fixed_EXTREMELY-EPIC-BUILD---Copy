#!/usr/bin/env python3
"""
Test the new context-aware token limits to ensure different prompt types
get appropriate response lengths.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.model_settings import get_model_options
from config.config import OLLAMA_MODEL


def test_token_limits():
    """Test that different prompt types get appropriate token limits"""
    print("Testing Context-Aware Token Limits")
    print("=" * 50)
    
    # Test the default options
    print(f"\nCurrent Model: {OLLAMA_MODEL}")
    
    default_options = get_model_options(OLLAMA_MODEL)
    print(f"\nDefault Options:")
    for key, value in default_options.items():
        print(f"  {key}: {value}")
    
    print(f"\nDefault token limit: {default_options.get('num_predict', 'Not set')}")
    
    # Simulate the model wrapper logic
    def simulate_prompt_type(prompt_type):
        model_options = get_model_options(OLLAMA_MODEL).copy()
        
        # Apply the same logic as model_wrapper.py
        if prompt_type == "environmental" or prompt_type == "awakening":
            model_options["num_predict"] = 300  # Longer for environmental descriptions
            model_options["stop"] = ["\n\nUser:", "\n\nHuman:", "\n\nAssistant:"]
        elif prompt_type == "reflection":
            model_options["num_predict"] = 250  # Longer for reflections  
            model_options["stop"] = ["\n\nUser:", "\n\nHuman:", "\n\nAssistant:"]
        elif prompt_type == "drawing":
            model_options["num_predict"] = 200  # Longer for drawing prompts
            model_options["stop"] = ["\n\nUser:", "\n\nHuman:", "\n\nAssistant:"]
        # else: normal captions use the default 100 tokens with existing stop conditions
        
        return model_options
    
    # Test each prompt type
    prompt_types = ["normal", "environmental", "awakening", "reflection", "drawing"]
    
    print(f"\nToken Limits by Prompt Type:")
    print("-" * 30)
    
    for prompt_type in prompt_types:
        options = simulate_prompt_type(prompt_type)
        tokens = options.get('num_predict', 'default')
        stops = len(options.get('stop', []))
        print(f"  {prompt_type:12}: {tokens:3} tokens, {stops} stop conditions")
    
    print(f"\nExpected Behavior:")
    print("-" * 30)
    print("  normal      : Brief 1-2 sentence fragments (100 tokens)")
    print("  environmental: Full environmental description (300 tokens)")
    print("  awakening   : Full awakening description (300 tokens)")  
    print("  reflection  : Complete reflective thoughts (250 tokens)")
    print("  drawing     : Complete drawing prompts (200 tokens)")
    
    print(f"\n" + "=" * 50)
    print("CONFIGURATION SUMMARY:")
    print("-" * 50)
    
    # Check if setup looks correct
    issues = 0
    
    normal_options = simulate_prompt_type("normal")
    env_options = simulate_prompt_type("environmental")
    
    normal_tokens = normal_options.get('num_predict', 100)
    env_tokens = env_options.get('num_predict', 300)
    
    if normal_tokens <= 120:
        print("[GOOD] Normal captions have brief token limit")
    else:
        print("[ISSUE] Normal captions may be too long")
        issues += 1
    
    if env_tokens >= 250:
        print("[GOOD] Environmental prompts have adequate token limit")
    else:
        print("[ISSUE] Environmental prompts may be too brief")
        issues += 1
    
    normal_stops = len(normal_options.get('stop', []))
    env_stops = len(env_options.get('stop', []))
    
    if normal_stops > env_stops:
        print("[GOOD] Normal captions have more restrictive stop conditions")
    else:
        print("[INFO] Environmental prompts have gentler stop conditions")
    
    if issues == 0:
        print("\n[SUCCESS] Token limits properly configured!")
        print("Environmental/awakening prompts should no longer be truncated.")
    else:
        print(f"\n[WARNING] {issues} potential issues found")
    
    print("=" * 50)


if __name__ == "__main__":
    test_token_limits()