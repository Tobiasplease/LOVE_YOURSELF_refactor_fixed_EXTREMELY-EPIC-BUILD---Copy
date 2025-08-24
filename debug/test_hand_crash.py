#!/usr/bin/env python3
"""
Test script to identify hand controller crashes
"""
import sys
import os
import time
import traceback
import threading

# Add the hand_control directory to Python path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'hand_control'))

def test_hand_interface_crash():
    """Test the hand control interface for crash issues."""
    print("Starting hand controller crash test...")
    
    try:
        # Import the hand controller
        from hand_control_interface import CleanCursorInterface
        print("SUCCESS: Imported hand control interface")
        
        # Test creating the interface in headless mode
        print("Creating hand control interface in headless mode...")
        interface = CleanCursorInterface(headless_mode=True)
        print("SUCCESS: Created hand control interface")
        
        # Let it run for 2 minutes to see if it crashes
        print("Running for 120 seconds to test for crashes...")
        start_time = time.time()
        
        while time.time() - start_time < 120:
            try:
                # Update the GUI event loop
                interface.root.update()
                time.sleep(0.1)
                
                # Print status every 10 seconds
                elapsed = time.time() - start_time
                if int(elapsed) % 10 == 0 and int(elapsed) > 0:
                    print(f"Running for {int(elapsed)} seconds - still alive")
                    
                    # Check for threading issues
                    print(f"Active threads: {threading.active_count()}")
                    for thread in threading.enumerate():
                        if thread != threading.current_thread():
                            print(f"  Thread: {thread.name} - alive: {thread.is_alive()}")
                            
            except Exception as e:
                print(f"ERROR during runtime: {e}")
                traceback.print_exc()
                break
        
        print("SUCCESS: Completed 120 second test without crash")
        
        # Cleanup
        interface.root.quit()
        interface.root.destroy()
        
    except Exception as e:
        print(f"ERROR: Hand controller test failed: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    test_hand_interface_crash()