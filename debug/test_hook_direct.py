#!/usr/bin/env python3
"""
Direct test to see what the fuck is going on with the hook system
"""

import os
import sys

# Add the project root to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

print("=== DIRECT HOOK TEST ===")

# 1. Check if hook module can be imported
try:
    import utils.hooks as hooks

    print(f"✅ utils.hooks imported successfully")
    print(f"   on_grbl_drawing_complete = {hooks.on_grbl_drawing_complete}")
except Exception as e:
    print(f"❌ Failed to import utils.hooks: {e}")
    sys.exit(1)

# 2. Check config values
try:
    from config.config import UARM_PLAY_AFTER_DRAW, UARM_PLAY_FILE, USE_UARM

    print(f"✅ Config imported successfully")
    print(f"   USE_UARM = {USE_UARM}")
    print(f"   UARM_PLAY_AFTER_DRAW = {UARM_PLAY_AFTER_DRAW}")
    print(f"   UARM_PLAY_FILE = {UARM_PLAY_FILE}")
    print(f"   File exists = {os.path.exists(UARM_PLAY_FILE) if UARM_PLAY_FILE else False}")
except Exception as e:
    print(f"❌ Failed to import config: {e}")
    sys.exit(1)

# 3. Test setting a simple hook
print("\n=== TESTING HOOK SETTING ===")


def test_hook():
    print("🔥 TEST HOOK WAS CALLED!")


hooks.on_grbl_drawing_complete = test_hook
print(f"✅ Hook set to: {hooks.on_grbl_drawing_complete}")

# 4. Test calling the hook
print("\n=== TESTING HOOK CALLING ===")
if callable(hooks.on_grbl_drawing_complete):
    print("✅ Hook is callable, calling it...")
    hooks.on_grbl_drawing_complete()
else:
    print(f"❌ Hook is not callable: {hooks.on_grbl_drawing_complete}")

# 5. Test the exact condition from machine.py
print("\n=== TESTING MACHINE.PY CONDITION ===")
condition = USE_UARM and UARM_PLAY_AFTER_DRAW and UARM_PLAY_FILE
print(f"Condition result: {condition}")
if condition:
    print("✅ Machine.py should register the hook")
else:
    print("❌ Machine.py will NOT register the hook")
    print(f"   USE_UARM: {USE_UARM}")
    print(f"   UARM_PLAY_AFTER_DRAW: {UARM_PLAY_AFTER_DRAW}")
    print(f"   UARM_PLAY_FILE: {UARM_PLAY_FILE}")

# 6. Simulate GRBL calling the hook
print("\n=== SIMULATING GRBL HOOK CALL ===")
try:
    from utils.hooks import on_grbl_drawing_complete

    print(f"Imported hook: {on_grbl_drawing_complete}")
    if callable(on_grbl_drawing_complete):
        print("🎯 CALLING HOOK LIKE GRBL DOES...")
        on_grbl_drawing_complete()
        print("✅ Hook call completed")
    else:
        print("❌ Hook not callable from GRBL perspective")
except Exception as e:
    print(f"❌ GRBL hook call failed: {e}")
    import traceback

    traceback.print_exc()

print("\n=== TEST COMPLETE ===")
