#!/usr/bin/env python3
"""
Test LCD brightness control
"""

import sys
import time

sys.path.append("/home/impostor/LOVE_YOURSELF_refactor_fixed_EXTREMELY-EPIC-BUILD---Copy")

from config.config import CAPTION_DISPLAY_PORT
from utils.caption_display import init_caption_display, send_caption_to_display, set_lcd_brightness


def test_brightness_control():
    """Test LCD brightness control with different levels."""
    print("Testing LCD brightness control...")

    # Initialize display
    init_caption_display(CAPTION_DISPLAY_PORT)
    time.sleep(2)

    # Test different brightness levels
    brightness_levels = [255, 128, 64, 32, 0, 128, 255]

    for brightness in brightness_levels:
        print(f"Setting brightness to {brightness}")
        set_lcd_brightness(brightness)
        send_caption_to_display(f"Brightness: {brightness}")
        time.sleep(3)

    print("Brightness test complete!")


def test_display_speed():
    """Test LCD display speed with long captions."""
    print("Testing LCD display speed...")

    # Initialize display
    init_caption_display(CAPTION_DISPLAY_PORT)
    time.sleep(2)

    # Test captions of different lengths
    captions = [
        "Short caption",
        "This is a medium length caption that should use both rows of the display",
        "This is a very long caption that will definitely require multiple chunks to display properly on the 16x2 LCD screen and should scroll through quickly",
        "Speed test complete!",
    ]

    for i, caption in enumerate(captions):
        print(f"Sending caption {i+1}: {caption[:50]}...")
        send_caption_to_display(caption)
        time.sleep(5)  # Watch it scroll

    print("Speed test complete!")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "speed":
        test_display_speed()
    else:
        test_brightness_control()
