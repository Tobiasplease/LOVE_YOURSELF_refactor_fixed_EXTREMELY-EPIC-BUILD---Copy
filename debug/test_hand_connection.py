#!/usr/bin/env python3
"""
Simple test to verify hand controller connection and startle
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from servo_control.hand_expression import HandExpressionController
import time

def test_connection():
    print("HAND CONTROLLER CONNECTION TEST")
    print("="*40)
    
    try:
        print("Attempting to connect to COM3...")
        controller = HandExpressionController(port="COM3", baudrate=9600)
        print("✅ Connection successful!")
        
        print("\nSending test commands...")
        
        # Send normal update (should appear in Arduino serial monitor)
        print("1. Normal consciousness update...")
        controller.update_from_consciousness(0.5, 0.3, 0.2, person_present=True)
        time.sleep(1)
        
        # Send startle command (should appear in Arduino serial monitor)
        print("2. Startle reaction...")
        controller.trigger_startle()
        time.sleep(2)
        
        # Another normal update
        print("3. Return to normal...")
        controller.update_from_consciousness(0.5, 0.3, 0.2, person_present=False)
        time.sleep(1)
        
        controller.close()
        print("✅ Test complete!")
        print("\nCheck Arduino Serial Monitor for:")
        print("- 'Consciousness command: xx,xx,xx,xx' messages")
        print("- Any error messages")
        
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        print("\nTroubleshooting:")
        print("1. Is Arduino connected to COM3?")
        print("2. Is the hand controller Arduino code uploaded?")
        print("3. Is another program using COM3?")

if __name__ == "__main__":
    test_connection()
