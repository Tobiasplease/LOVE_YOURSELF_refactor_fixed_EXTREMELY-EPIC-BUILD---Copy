#!/usr/bin/env python3
"""
Test script to verify brevity settings are properly configured.
Shows the token limits and stop conditions for caption generation.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.model_settings import get_model_options
from config.config import OLLAMA_MODEL
from config.prompt_templates import SYSTEM_PROMPT


def test_brevity_settings():
    """Test that brevity settings are configured correctly"""
    print("Testing Brevity Settings")
    print("=" * 50)
    
    # Check current model settings
    print(f"\nCurrent Model: {OLLAMA_MODEL}")
    
    # Get model options
    options = get_model_options(OLLAMA_MODEL)
    print(f"\nModel Options:")
    for key, value in options.items():
        print(f"  {key}: {value}")
    
    # Check token limit
    num_predict = options.get('num_predict', 'Not set')
    print(f"\nToken Limit Analysis:")
    print(f"  num_predict: {num_predict}")
    
    if isinstance(num_predict, int):
        if num_predict <= 120:
            print(f"  ✓ GOOD: Token limit is {num_predict} - should encourage brief responses")
        elif num_predict <= 200:
            print(f"  ⚠ WARNING: Token limit is {num_predict} - may allow medium-length responses")
        else:
            print(f"  ✗ TOO HIGH: Token limit is {num_predict} - will allow long monologues")
    
    # Check stop conditions
    stop_conditions = options.get('stop', [])
    print(f"\nStop Conditions ({len(stop_conditions)} total):")
    for stop in stop_conditions:
        print(f"  - \"{stop}\"")
    
    # Check system prompt brevity instructions
    print(f"\nSystem Prompt Brevity Instructions:")
    if "1-2 brief sentences" in SYSTEM_PROMPT:
        print("  ✓ GOOD: Contains '1-2 brief sentences' instruction")
    elif "2-3 short sentences" in SYSTEM_PROMPT:
        print("  ⚠ OKAY: Contains '2-3 short sentences' instruction")
    else:
        print("  ✗ MISSING: No clear brevity instruction found")
    
    if "fragment" in SYSTEM_PROMPT.lower():
        print("  ✓ GOOD: Contains fragmented thought instruction")
    else:
        print("  ⚠ MISSING: No fragmented thought instruction")
    
    if "unfold across multiple" in SYSTEM_PROMPT:
        print("  ✓ GOOD: Contains stream-of-consciousness instruction")
    else:
        print("  ⚠ MISSING: No stream-of-consciousness instruction")
    
    print("\n" + "=" * 50)
    print("SUMMARY:")
    print("-" * 50)
    
    # Overall assessment
    issues = 0
    
    if isinstance(num_predict, int) and num_predict > 150:
        print("❌ Token limit too high - will generate long responses")
        issues += 1
    else:
        print("✅ Token limit appropriately set for brief responses")
    
    if "1-2 brief sentences" in SYSTEM_PROMPT:
        print("✅ System prompt enforces brevity")
    else:
        print("❌ System prompt brevity instructions could be stronger")
        issues += 1
    
    if issues == 0:
        print("\n🎉 All brevity settings correctly configured!")
        print("Expected behavior: 1-2 sentence fragments that build a stream of consciousness")
    else:
        print(f"\n⚠️ {issues} issues found - may still generate long responses")
    
    print("=" * 50)


if __name__ == "__main__":
    test_brevity_settings()