#!/usr/bin/env python3
"""
Test if GRBL completion actually calls the hook
"""

import os
import sys

# Add the project root to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

print("=== TESTING GRBL HOOK CALL ===")

# Set up a test hook
try:
    import utils.hooks as hooks

    def test_grbl_hook():
        print("🎯 GRBL HOOK WAS CALLED BY GRBL!")

    hooks.on_grbl_drawing_complete = test_grbl_hook
    print(f"✅ Test hook set up: {hooks.on_grbl_drawing_complete}")

except Exception as e:
    print(f"❌ Failed to set up test hook: {e}")
    sys.exit(1)

# Now simulate what GRBL does in grbl_utils.py
print("\n=== SIMULATING GRBL COMPLETION ===")

try:
    from utils.hooks import on_grbl_drawing_complete

    print(f"GRBL perspective - hook = {on_grbl_drawing_complete}")

    if callable(on_grbl_drawing_complete):
        print("✅ Hook is callable from GRBL")
        print("🎯 Calling hook like GRBL does...")
        on_grbl_drawing_complete()
        print("✅ Hook call completed")
    else:
        print(f"❌ Hook not callable: {on_grbl_drawing_complete}")

except Exception as e:
    print(f"❌ GRBL hook call failed: {e}")
    import traceback

    traceback.print_exc()

print("\n=== TEST COMPLETE ===")
