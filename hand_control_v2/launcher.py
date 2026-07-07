#!/usr/bin/env python3
"""
Consolidated Hand Controller Launcher
======================================
Standalone launcher for testing the unified hand + arm servo controller.
Run this directly to test the UI and recording system independently of machine.py.
"""

import os
import sys

# Add parent directory for config imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Run from the hand_control_v2 directory so dataset paths resolve correctly
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from hand_control_interface import CleanCursorInterface


def main():
    print("Starting Consolidated Hand Controller (v2)...")
    print("  - 4 finger servos (wave/cursor control)")
    print("  - 2 arm servos (slider control)")
    print("  - Unified recording into 6D Markov chains")
    print()

    interface = CleanCursorInterface()

    def on_closing():
        print("Shutting down...")
        interface.cleanup_all_timers()
        interface.root.destroy()

    interface.root.protocol("WM_DELETE_WINDOW", on_closing)
    interface.root.mainloop()


if __name__ == "__main__":
    main()
