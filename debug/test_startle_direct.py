#!/usr/bin/env python3
"""
Test the startle reaction directly with the hand controller
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from servo_control.hand_expression import HandExpressionController
import time

def test_startle():
    print("STARTLE REACTION TEST")
    print("====================")
    
    try:
        print("Connecting to hand controller...")
        controller = HandExpressionController(port="COM3", baudrate=9600)
        print("✅ Hand controller connected!")
        
        print("\nStarting normal operation for 3 seconds...")
        for i in range(30):
            controller.update_from_consciousness(0.5, 0.3, 0.2)  # neutral state
            time.sleep(0.1)
            
        print("\n💥 TRIGGERING STARTLE REACTION!")
        controller.trigger_startle()
        
        print("Watching startle reaction for 2 seconds...")
        for i in range(20):
            # Don't send new commands during startle - let it complete
            time.sleep(0.1)
            
        print("✅ Startle reaction complete!")
        
        print("\nResuming normal operation...")
        for i in range(20):
            controller.update_from_consciousness(0.5, 0.3, 0.2)  # back to neutral
            time.sleep(0.1)
            
        controller.close()
        print("✅ Test complete!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        if 'controller' in locals():
            controller.close()

if __name__ == "__main__":
    test_startle()
