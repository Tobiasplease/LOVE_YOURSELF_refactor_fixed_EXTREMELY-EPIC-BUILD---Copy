#!/usr/bin/env python3
"""
Test the fixed drawing state detection that checks both ComfyUI and GRBL phases
"""

import os
import sys
import time

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from captioner.captioner import Captioner
from utils.state_manager import state_manager


def test_drawing_state_detection_fix():
    """Test that drawing state detection works for both ComfyUI and GRBL phases."""
    print("🧪 Testing fixed drawing state detection...")

    captioner = Captioner()

    # Test 1: No drawing activity
    state_manager.is_generating_drawing = False
    state_manager.is_executing_cnc = False

    is_drawing = captioner._is_currently_drawing()
    print(f"   No activity: {not is_drawing} ✓")
    assert not is_drawing, "Should detect no drawing activity"

    # Test 2: ComfyUI generation phase only
    state_manager.is_generating_drawing = True
    state_manager.is_executing_cnc = False
    state_manager.drawing_start_time = time.time()
    state_manager.current_drawing_prompt = "Test ComfyUI prompt"

    is_drawing = captioner._is_currently_drawing()
    print(f"   ComfyUI generating: {is_drawing} ✓")
    assert is_drawing, "Should detect ComfyUI generation phase"

    # Test 3: GRBL execution phase only
    state_manager.is_generating_drawing = False
    state_manager.is_executing_cnc = True

    is_drawing = captioner._is_currently_drawing()
    print(f"   GRBL executing: {is_drawing} ✓")
    assert is_drawing, "Should detect GRBL execution phase"

    # Test 4: Both phases active (transition period)
    state_manager.is_generating_drawing = True
    state_manager.is_executing_cnc = True

    is_drawing = captioner._is_currently_drawing()
    print(f"   Both phases active: {is_drawing} ✓")
    assert is_drawing, "Should detect when both phases are active"

    # Clean up
    state_manager.is_generating_drawing = False
    state_manager.is_executing_cnc = False
    state_manager.drawing_start_time = None
    state_manager.current_drawing_prompt = None

    print("✅ Drawing state detection fix working correctly!\n")
    print("This should prevent multiple drawing triggers because:")
    print("- ComfyUI generation will now trigger introspection mode")
    print("- Introspection mode prevents visual captioning")
    print("- No visual captioning = no additional drawing triggers")
    print("- System will stay in introspection until drawing fully complete")


if __name__ == "__main__":
    test_drawing_state_detection_fix()
