#!/usr/bin/env python3
"""
Test the improved LCD display system.
Tests the new state tracking and caption skipping logic.
"""

import time
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.caption_display import init_caption_display, send_caption_to_display


def test_lcd_display():
    """Test LCD display with rapid captions to verify skipping behavior."""
    print("🖥️  Starting LCD display test...")

    # Initialize display
    init_caption_display()
    time.sleep(2)

    # Test rapid captions (should skip appropriately)
    captions = [
        "First caption - this should display immediately",
        "Second caption - this should be pending or skipped",
        "Third caption - this should replace pending",
        "Fourth caption - most recent should win",
        "Very long caption that will need to scroll across multiple chunks because it's much longer than the 32 character display width",
        "Short final caption",
    ]

    print("\n📝 Sending test captions rapidly...")
    for i, caption in enumerate(captions):
        print(f"\n[TEST] Sending caption {i+1}: {caption[:40]}...")
        send_caption_to_display(caption)
        time.sleep(0.5)  # Rapid sending

    print("\n⏳ Waiting to see final display behavior...")
    time.sleep(10)

    # Test with longer delays
    print("\n📝 Testing with proper delays...")
    time.sleep(3)
    send_caption_to_display("Caption after long delay - should display immediately")
    time.sleep(5)
    send_caption_to_display("Final test caption - checking continuous updates")

    print("\n✅ Test complete!")


if __name__ == "__main__":
    test_lcd_display()