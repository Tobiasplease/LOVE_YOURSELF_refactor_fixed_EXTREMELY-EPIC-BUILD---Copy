#!/usr/bin/env python3
"""Test COM port connection directly"""
import sys
import os
import traceback

# Add the hand_control directory to Python path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'hand_control'))

def test_com_connection():
    """Test direct COM port connection."""
    print("Testing COM port connection...")
    
    try:
        from hand_expression import HandExpressionController
        print("SUCCESS: Imported HandExpressionController")
        
        # Test COM3 connection with verbose output
        print("\nTesting COM3 connection...")
        try:
            controller = HandExpressionController(port="COM3", clean_output=False)
            
            # The connection is lazy, so we need to trigger it
            print("Triggering connection by setting hand positions...")
            controller.set_hand_positions([45, 90, 135, 90])
            
            if controller.serial_connection:
                print("SUCCESS: Connected to COM3")
                print("SUCCESS: Commands sent successfully")
                controller.cleanup()
            else:
                print("ERROR: Failed to connect to COM3")
        except Exception as e:
            print(f"ERROR connecting to COM3: {e}")
            traceback.print_exc()
            
        # Test COM16 connection 
        print("\nTesting COM16 connection...")
        try:
            controller2 = HandExpressionController(port="COM16", clean_output=False)
            
            # The connection is lazy, so we need to trigger it
            print("Triggering connection by setting hand positions...")
            controller2.set_hand_positions([45, 90, 135, 90])
            
            if controller2.serial_connection:
                print("SUCCESS: Connected to COM16")
                print("SUCCESS: Commands sent successfully")
                controller2.cleanup()
            else:
                print("ERROR: Failed to connect to COM16")
        except Exception as e:
            print(f"ERROR connecting to COM16: {e}")
            traceback.print_exc()
            
    except Exception as e:
        print(f"ERROR: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    test_com_connection()