#!/usr/bin/env python3

"""Test the improved caption flow system with Arduino feedback."""

import time
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.caption_display import CaptionDisplay

def test_caption_flow():
    print("Testing improved caption flow system...")

    # Initialize caption display
    try:
        display = CaptionDisplay(port="/dev/ttyUSB0")
        if not display.connected:
            print("Failed to connect to Arduino - please check connection")
            return

        print("Connected to Arduino. Testing caption queue system...")

        # Test captions of varying lengths
        test_captions = [
            "Short caption",
            "This is a medium length caption that should span multiple chunks on the LCD display",
            "Quick update",
            "Another very long caption that definitely requires multiple display chunks to show completely on the single-row LCD",
            "Final test",
        ]

        print("Sending test captions rapidly to test queue system...")

        for i, caption in enumerate(test_captions):
            print(f"Sending caption {i+1}: {caption[:30]}...")
            display.send_caption(caption)
            time.sleep(0.2)  # Send quickly to test queue

        print("Captions sent. Monitor LCD for continuous flow.")
        print("Press Ctrl+C to stop...")

        # Keep the script running to monitor
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nStopping test...")

    except Exception as e:
        print(f"Test error: {e}")
    finally:
        try:
            display.close()
        except:
            pass

if __name__ == "__main__":
    test_caption_flow()