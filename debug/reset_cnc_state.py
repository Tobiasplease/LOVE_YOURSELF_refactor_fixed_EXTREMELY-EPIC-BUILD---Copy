#!/usr/bin/env python3
"""
Reset CNC Execution State
========================

Quick utility to reset stuck CNC execution state when the system
gets stuck at "executing: True" but no drawing is actually happening.

Usage: python debug/reset_cnc_state.py
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from utils.state_manager import state_manager

    print("[🔧] Checking CNC execution state...")

    is_executing = getattr(state_manager, "is_executing_cnc", False)
    cnc_file = getattr(state_manager, "current_gcode_file", None)
    cnc_phase = getattr(state_manager, "current_drawing_phase", None)

    print(f"Current state:")
    print(f"  - is_executing_cnc: {is_executing}")
    print(f"  - current_gcode_file: {cnc_file}")
    print(f"  - current_drawing_phase: {cnc_phase}")

    if is_executing:
        print("\n[⚠️] CNC execution state is TRUE - clearing it...")
        state_manager.finish_cnc_execution()
        print("[✅] CNC execution state cleared!")
        print("\nDrawing system should now be able to start new drawings.")
    else:
        print("\n[✅] CNC execution state is already clear.")

    print("\nDone!")

except ImportError as e:
    print(f"[❌] Could not import state_manager: {e}")
    print("Make sure you're running this from the project root directory.")
except Exception as e:
    print(f"[❌] Error: {e}")
    import traceback
    traceback.print_exc()