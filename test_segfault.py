#!/usr/bin/env python3
"""Quick test script to simulate a segfault for testing the restart wrapper."""

import ctypes
import sys

def cause_segfault():
    """Deliberately cause a segmentation fault."""
    print("Causing segfault in 3 seconds...")
    import time
    time.sleep(3)

    # This will cause a segfault
    ctypes.string_at(0)

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--segfault":
        cause_segfault()
    else:
        print("Normal exit")