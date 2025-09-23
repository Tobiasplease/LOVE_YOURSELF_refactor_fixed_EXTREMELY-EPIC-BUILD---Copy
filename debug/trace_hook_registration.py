#!/usr/bin/env python3
"""
Trace exactly what happens during hook registration in machine.py
"""

import os
import sys

# Add the project root to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

print("=== HOOK REGISTRATION TRACE ===")

# 1. Test the exact same imports as machine.py
print("\n1. Testing imports...")
try:
    import utils.hooks as _hooks

    print(f"✅ utils.hooks imported: {_hooks}")
    print(f"   Initial hook value: {_hooks.on_grbl_drawing_complete}")
except Exception as e:
    print(f"❌ utils.hooks import failed: {e}")
    sys.exit(1)

try:
    from config.config import UARM_PLAY_AFTER_DRAW, UARM_PLAY_FILE, USE_UARM

    print(f"✅ Config imports successful")
    print(f"   USE_UARM: {USE_UARM}")
    print(f"   UARM_PLAY_AFTER_DRAW: {UARM_PLAY_AFTER_DRAW}")
    print(f"   UARM_PLAY_FILE: {UARM_PLAY_FILE}")
except Exception as e:
    print(f"❌ Config import failed: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)

# 2. Test the condition
print(f"\n2. Testing condition...")
condition = USE_UARM and UARM_PLAY_AFTER_DRAW and UARM_PLAY_FILE
print(f"   Condition result: {condition}")
print(f"   USE_UARM: {USE_UARM} (type: {type(USE_UARM)})")
print(f"   UARM_PLAY_AFTER_DRAW: {UARM_PLAY_AFTER_DRAW} (type: {type(UARM_PLAY_AFTER_DRAW)})")
print(f"   UARM_PLAY_FILE: {UARM_PLAY_FILE} (type: {type(UARM_PLAY_FILE)})")

if not condition:
    print("❌ Condition failed - hook will NOT be registered")
    sys.exit(1)

# 3. Test hook registration (like machine.py does)
print(f"\n3. Testing hook registration...")


def test_uarm_hook():
    print("🎯 UARM HOOK WAS CALLED!")


try:
    _hooks.on_grbl_drawing_complete = test_uarm_hook
    print(f"✅ Hook registered: {_hooks.on_grbl_drawing_complete}")
except Exception as e:
    print(f"❌ Hook registration failed: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)

# 4. Test if hook is callable from GRBL perspective
print(f"\n4. Testing GRBL hook call...")
try:
    from utils.hooks import on_grbl_drawing_complete

    print(f"   Imported hook: {on_grbl_drawing_complete}")
    if callable(on_grbl_drawing_complete):
        print("✅ Hook is callable from GRBL")
        on_grbl_drawing_complete()
    else:
        print(f"❌ Hook not callable: {on_grbl_drawing_complete}")
except Exception as e:
    print(f"❌ GRBL hook call failed: {e}")
    import traceback

    traceback.print_exc()

# 5. Check if machine.py is actually running the registration block
print(f"\n5. Checking if machine.py registration would work...")

# Simulate exactly what machine.py does
try:
    print(f"[DEBUG] Hook setup: USE_UARM={USE_UARM}, UARM_PLAY_AFTER_DRAW={UARM_PLAY_AFTER_DRAW}, UARM_PLAY_FILE={UARM_PLAY_FILE}")

    if USE_UARM and UARM_PLAY_AFTER_DRAW and UARM_PLAY_FILE:

        def _normalize_smooth(path: str) -> str:
            base, ext = os.path.splitext(path)
            while base.lower().endswith(".smooth"):
                base = base[:-7]
            return f"{base}{ext or '.txt'}"

        def _uarm_after_grbl():
            print(f"[DEBUG] _uarm_after_grbl() called - starting papermove sequence")
            print("[SIMULATION] Would run papermove here")

        _hooks.on_grbl_drawing_complete = _uarm_after_grbl
        print(f"[DEBUG] GRBL hook registered successfully - uArm will trigger after drawing completion")

        # Test the registered hook
        print("\n6. Testing registered hook...")
        _hooks.on_grbl_drawing_complete()

    else:
        print(f"[DEBUG] GRBL hook NOT registered - condition failed")

except Exception as e:
    print(f"[DEBUG] GRBL hook registration failed: {e}")
    import traceback

    traceback.print_exc()

print("\n=== TRACE COMPLETE ===")
