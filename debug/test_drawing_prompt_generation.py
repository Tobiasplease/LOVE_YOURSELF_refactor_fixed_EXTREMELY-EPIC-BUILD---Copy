#!/usr/bin/env python3
"""
Test script to isolate which method in build_drawing_prompt is hanging.
"""

import sys
import os
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from captioner.captioner import Captioner
from captioner.prompts import build_drawing_prompt

def test_memory_methods():
    """Test each memory method individually to find which one hangs."""
    print("Creating Captioner instance...")
    captioner = Captioner()

    # Set some basic state
    captioner.last_caption = "Test caption for drawing"
    captioner.current_mood = 0.5

    print("\n=== Testing individual memory methods ===")

    # Test get_recent_memory
    print("1. Testing get_recent_memory()...")
    try:
        start_time = time.time()
        memory = captioner.get_recent_memory()
        elapsed = time.time() - start_time
        print(f"   ✓ Success in {elapsed:.2f}s: {memory[:50]}...")
    except Exception as e:
        print(f"   ✗ Failed: {e}")

    # Test get_last_reflection
    print("2. Testing get_last_reflection()...")
    try:
        start_time = time.time()
        reflection = captioner.get_last_reflection()
        elapsed = time.time() - start_time
        print(f"   ✓ Success in {elapsed:.2f}s: {reflection[:50]}...")
    except Exception as e:
        print(f"   ✗ Failed: {e}")

    # Test describe_current_mood
    print("3. Testing describe_current_mood()...")
    try:
        start_time = time.time()
        mood = captioner.describe_current_mood()
        elapsed = time.time() - start_time
        print(f"   ✓ Success in {elapsed:.2f}s: {mood}")
    except Exception as e:
        print(f"   ✗ Failed: {e}")

    # Test get_top_motifs (most likely culprit)
    print("4. Testing get_top_motifs()...")
    try:
        start_time = time.time()
        motifs = captioner.get_top_motifs(6)
        elapsed = time.time() - start_time
        print(f"   ✓ Success in {elapsed:.2f}s: {motifs}")
    except Exception as e:
        print(f"   ✗ Failed: {e}")

    return captioner

def test_drawing_prompt_generation():
    """Test the full drawing prompt generation."""
    print("\n=== Testing full build_drawing_prompt ===")

    captioner = test_memory_methods()

    print("5. Testing build_drawing_prompt()...")
    try:
        start_time = time.time()
        prompt = build_drawing_prompt(captioner, extra="Test extra context")
        elapsed = time.time() - start_time
        print(f"   ✓ Success in {elapsed:.2f}s")
        print(f"   Prompt length: {len(prompt)} characters")
        print(f"   Prompt preview: {prompt[:100]}...")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        import traceback
        traceback.print_exc()

def test_model_drawing_prompt():
    """Test the model's generate_drawing_prompt method."""
    print("\n=== Testing model generate_drawing_prompt ===")

    captioner = Captioner()
    captioner.last_caption = "Test caption for drawing"

    print("6. Testing model.generate_drawing_prompt()...")
    try:
        start_time = time.time()
        prompt = captioner.model.generate_drawing_prompt(extra="Test context")
        elapsed = time.time() - start_time
        print(f"   ✓ Success in {elapsed:.2f}s")
        print(f"   Generated prompt: {prompt[:100]}...")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("🧪 Testing Drawing Prompt Generation")
    print("=" * 50)

    test_drawing_prompt_generation()
    test_model_drawing_prompt()

    print("\n✅ Test completed!")