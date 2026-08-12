#!/usr/bin/env python3
"""
Test caption continuation to see if we get flowing thoughts instead of declarative statements.
"""
import os
import sys
import time

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unittest.mock import Mock

from captioner.captioner import Captioner


def create_mock_frame():
    """Create a mock frame for testing."""
    import numpy as np

    return np.zeros((480, 640, 3), dtype=np.uint8)


def main():
    print("🧪 Testing Caption Continuation System")
    print("=" * 60)

    # Create captioner
    captioner = Captioner()

    # Manually set some state to simulate ongoing session
    captioner.true_session_start = time.time() - 300  # 5 minutes ago

    # Set up a sequence of thoughts to test continuation
    test_thoughts = [
        "The mechanical workspace spreads before me, cluttered with servo motors and drawing equipment that sparks my technical curiosity.",
        # Next should continue from "sparks my technical curiosity" without starting fresh
    ]

    # First caption
    frame = create_mock_frame()
    first_caption = captioner.caption_frame(frame, False, None)
    print(f"First caption: {first_caption}")

    # Wait a moment then get second caption - this should continue the thought
    time.sleep(1)
    second_caption = captioner.caption_frame(frame, False, None)
    print(f"\nSecond caption: {second_caption}")

    # Check for continuation vs new observation
    print(f"\n=== ANALYSIS ===")

    # Check if second caption starts with declarative statements
    declarative_starts = ["as i", "in my", "i am", "i can see", "looking at", "observing", "from my"]
    is_declarative = any(second_caption.lower().startswith(start) for start in declarative_starts)

    if is_declarative:
        print("❌ Second caption starts with declarative statement")
        print(f"   Starts with: '{second_caption[:50]}...'")
    else:
        print("✅ Second caption appears to continue thought")
        print(f"   Starts with: '{second_caption[:50]}...'")

    # Check for semantic connection
    first_words = set(first_caption.lower().split())
    second_words = set(second_caption.lower().split()[:10])  # First 10 words
    shared_concepts = first_words.intersection(second_words)

    if len(shared_concepts) > 2:
        print(f"✅ Semantic connection found: {shared_concepts}")
    else:
        print(f"❌ Limited semantic connection: {shared_concepts}")


if __name__ == "__main__":
    main()
