#!/usr/bin/env python3
"""
Debug what the GUI is actually sending to the Arduino
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

print("=== GUI COMMAND DEBUG ===")
print(f"DETECTED_HAND_PORT: {os.environ.get('DETECTED_HAND_PORT')}")

class DebugHandController:
    """Wrapper to monitor what commands are sent"""
    def __init__(self):
        self.commands_sent = []
        self.connected = False
        
    def log_command(self, method_name, *args, **kwargs):
        timestamp = time.time()
        self.commands_sent.append((timestamp, method_name, args, kwargs))
        print(f"[{timestamp:.2f}] GUI CALLED: {method_name}({args}, {kwargs})")

# Override the HandExpressionController to monitor calls
original_controller = None

def debug_gui():
    global original_controller
    try:
        from hand_control.hand_control_interface import CleanCursorInterface
        from hand_control.hand_expression import HandExpressionController
        
        # Monkey patch to monitor commands
        original_init = HandExpressionController.__init__
        original_set_hand_positions = HandExpressionController.set_hand_positions
        original_enable_manual = HandExpressionController.enable_manual_override
        
        debug_controller = DebugHandController()
        
        def debug_init(self, *args, **kwargs):
            debug_controller.log_command("__init__", *args, **kwargs)
            result = original_init(self, *args, **kwargs)
            debug_controller.connected = hasattr(self, 'serial_connection') and self.serial_connection is not None
            print(f"[DEBUG] HandController connection status: {debug_controller.connected}")
            return result
            
        def debug_set_hand_positions(self, positions):
            debug_controller.log_command("set_hand_positions", positions)
            return original_set_hand_positions(self, positions)
            
        def debug_enable_manual(self):
            debug_controller.log_command("enable_manual_override")
            return original_enable_manual(self)
        
        # Apply patches
        HandExpressionController.__init__ = debug_init
        HandExpressionController.set_hand_positions = debug_set_hand_positions
        HandExpressionController.enable_manual_override = debug_enable_manual
        
        print("Creating GUI with debug monitoring...")
        controller = CleanCursorInterface()
        
        # Let GUI initialize
        time.sleep(3)
        
        print(f"\n=== DEBUG RESULTS AFTER 3 SECONDS ===")
        print(f"Commands sent to Arduino: {len(debug_controller.commands_sent)}")
        for timestamp, method, args, kwargs in debug_controller.commands_sent:
            print(f"  [{timestamp:.2f}] {method}({args}, {kwargs})")
        
        if hasattr(controller, 'connected'):
            print(f"GUI connected status: {controller.connected}")
            
        if hasattr(controller, 'hand_controller') and controller.hand_controller:
            print(f"HandController exists: True")
            if hasattr(controller.hand_controller, 'serial_connection'):
                print(f"Serial connection exists: {controller.hand_controller.serial_connection is not None}")
        else:
            print("HandController exists: False")
            
        # Test manual command sending
        print("\n=== TESTING MANUAL COMMAND ===")
        if hasattr(controller, 'hand_controller') and controller.hand_controller:
            print("Sending test command: [90, 45, 135, 90]")
            controller.hand_controller.set_hand_positions([90, 45, 135, 90])
            time.sleep(1)
            
        # Wait and close
        print("\nClosing GUI after test...")
        controller.root.after(2000, controller.root.quit)
        controller.root.mainloop()
        
        # Restore original methods
        HandExpressionController.__init__ = original_init
        HandExpressionController.set_hand_positions = original_set_hand_positions  
        HandExpressionController.enable_manual_override = original_enable_manual
        
        print(f"\n=== FINAL COMMAND COUNT ===")
        print(f"Total commands sent: {len(debug_controller.commands_sent)}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_gui()