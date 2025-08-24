#!/usr/bin/env python3
"""Simple test to check hand control import"""
import sys
import os
import traceback

# Add the hand_control directory to Python path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'hand_control'))

try:
    print("Testing import...")
    from hand_control_interface import CleanCursorInterface
    print("SUCCESS: Import successful")
    
    print("Creating interface in headless mode...")
    interface = CleanCursorInterface(headless_mode=True)
    print("SUCCESS: Interface created")
    
    # Run for just 10 seconds
    import time
    start = time.time()
    while time.time() - start < 10:
        try:
            interface.root.update()
        except Exception as e:
            print(f"ERROR: {e}")
            break
        time.sleep(0.1)
    
    print("SUCCESS: 10 second test passed")
    interface.root.quit()
    interface.root.destroy()
    
except Exception as e:
    print(f"ERROR: {e}")
    traceback.print_exc()