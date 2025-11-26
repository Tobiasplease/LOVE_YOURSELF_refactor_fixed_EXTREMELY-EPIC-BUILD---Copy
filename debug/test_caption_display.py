#!/usr/bin/env python3
"""Test script for LCD caption display functionality."""

import time

from config.config import CAPTION_DISPLAY_PORT
from utils.caption_display import close_caption_display, init_caption_display, send_caption_to_display


def test_caption_display():
    print(f"Testing LCD caption display on {CAPTION_DISPLAY_PORT}")

    # Initialize display
    init_caption_display(CAPTION_DISPLAY_PORT)
    time.sleep(2)

    # Test captions
    test_captions = [
        "Hello World!",
        "This is a longer caption that should scroll across the display nicely",
        "Short text",
        "AI consciousness awakening... observing the environment with wonder and curiosity.",
        "Testing emoji symbols: * # @ & % ! ?",
        "Numbers: 123456789 Temperature: 25.5C Time: 14:32",
        "Final test complete",
    ]

    print("Sending test captions...")
    for i, caption in enumerate(test_captions):
        print(f"Caption {i+1}: {caption[:50]}...")
        send_caption_to_display(caption)
        time.sleep(5)  # Wait to see scrolling

    print("Test complete!")
    close_caption_display()


if __name__ == "__main__":
    try:
        test_caption_display()
    except KeyboardInterrupt:
        print("\nTest interrupted")
        close_caption_display()
    except Exception as e:
        print(f"Test failed: {e}")
        close_caption_display()
