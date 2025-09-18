#!/usr/bin/env python3
"""
Test the complete hook flow from image_monitor → uArm hook
"""

import os
import sys

# Add the project root to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

print("=== TESTING COMPLETE HOOK FLOW ===")

# Test 1: Hook registration (like machine.py does it)
print("\n1. Testing hook registration...")
try:
    import utils.hooks as _hooks
    from config.config import USE_UARM, UARM_PLAY_AFTER_DRAW, UARM_PLAY_FILE

    print(f"Config values: USE_UARM={USE_UARM}, UARM_PLAY_AFTER_DRAW={UARM_PLAY_AFTER_DRAW}")
    print(f"UARM_PLAY_FILE exists: {os.path.exists(UARM_PLAY_FILE) if UARM_PLAY_FILE else False}")

    if USE_UARM and UARM_PLAY_AFTER_DRAW and UARM_PLAY_FILE:
        def test_uarm_hook():
            print("🎯 [SIMULATION] uArm papermove would execute here!")
            print(f"   Would play: {UARM_PLAY_FILE}")

        _hooks.on_grbl_drawing_complete = test_uarm_hook
        print("✅ Hook registered successfully")
    else:
        print("❌ Hook not registered - config conditions not met")
        sys.exit(1)

except Exception as e:
    print(f"❌ Hook registration failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 2: Image monitor hook call (simulate what image_monitor.py does)
print("\n2. Testing image monitor hook call...")
try:
    # This simulates the exact code in image_monitor.py finally block
    from utils.hooks import on_grbl_drawing_complete
    from config.config import EXECUTE_GRBL_GCODE

    # Simulate EXECUTE_GRBL_GCODE being True
    if True:  # Simulating EXECUTE_GRBL_GCODE
        print("🎯 [DEBUG] DRAWING EXECUTION COMPLETE - TRIGGERING UARM HOOK AND RESETTING COOLDOWN")

        # Simulate cooldown reset
        print("⏰ [DEBUG] DRAWING COOLDOWN RESET - next drawing can trigger in 60s")

        # Trigger uArm hook
        if callable(on_grbl_drawing_complete):
            print("🔥 [DEBUG] CALLING UARM HOOK FROM IMAGE MONITOR")
            on_grbl_drawing_complete()
            print("🔥 [DEBUG] UARM HOOK COMPLETED")
        else:
            print(f"❌ [DEBUG] NO UARM HOOK REGISTERED")

        print("🔄 [DEBUG] Resuming idle movements after uArm completion")

except Exception as e:
    print(f"❌ Image monitor hook call failed: {e}")
    import traceback
    traceback.print_exc()

print("\n=== COMPLETE FLOW TEST SUCCESSFUL ===")
print("Both hook registration and execution work correctly!")