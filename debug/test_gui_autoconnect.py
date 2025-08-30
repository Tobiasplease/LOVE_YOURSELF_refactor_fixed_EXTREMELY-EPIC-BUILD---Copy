#!/usr/bin/env python3
"""
Test if GUI auto-connects now
"""
import os
import sys
import time
import threading
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set environment variables
import arduino_port_detector
detector = arduino_port_detector.ArduinoPortDetector()
detector.set_environment_variables()

print("=== GUI AUTO-CONNECT TEST ===")
print(f"DETECTED_HAND_PORT: {os.environ.get('DETECTED_HAND_PORT')}")

try:
    from hand_control.hand_control_interface import CleanCursorInterface
    
    # Create GUI in background thread to capture output
    gui_output = []
    
    def capture_print(msg):
        gui_output.append(str(msg))
        print(f"[GUI] {msg}")
    
    # Override print temporarily
    original_print = print
    
    def run_gui():
        try:
            # Create GUI
            controller = CleanCursorInterface()
            
            # Check connection status after initialization
            time.sleep(2)  # Give auto-connect time to work
            
            if hasattr(controller, 'connected') and controller.connected:
                print("✅ SUCCESS: GUI auto-connected!")
                print(f"   Status: {controller.status_label.cget('text')}")
            else:
                print("❌ FAILED: GUI did not auto-connect")
                print(f"   Status: {controller.status_label.cget('text')}")
            
            # Close GUI after test
            controller.root.after(1000, controller.root.quit)
            controller.root.mainloop()
            
        except Exception as e:
            print(f"❌ GUI test failed: {e}")
    
    # Run GUI test
    print("Starting GUI auto-connect test...")
    print("(GUI will close automatically after test)")
    
    gui_thread = threading.Thread(target=run_gui, daemon=True)
    gui_thread.start()
    gui_thread.join(timeout=10)  # 10 second timeout
    
    print("GUI test completed")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()