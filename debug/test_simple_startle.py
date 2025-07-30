#!/usr/bin/env python3
"""
Simple debug script to test if face detection startle is working
"""

import cv2
import time
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from servo_control.hand_expression import HandExpressionController

def main():
    print("SIMPLE STARTLE TEST")
    print("="*40)
    print("Testing startle reaction every 3 seconds")
    print("You should see quick finger snaps!")
    print("Press Ctrl+C to stop")
    print()
    
    try:
        hand_controller = HandExpressionController()
        print("✅ Hand controller connected")
    except Exception as e:
        print(f"❌ Hand controller failed: {e}")
        return
    
    test_count = 0
    while True:
        try:
            test_count += 1
            print(f"\n🎯 TEST {test_count}: Triggering startle reaction...")
            
            # Trigger startle
            hand_controller.trigger_startle()
            print("✅ Startle command sent!")
            
            # Wait and show countdown
            for i in range(3, 0, -1):
                print(f"   Next test in {i} seconds...", end='\r')
                time.sleep(1)
            print()  # Clear the line
            
        except KeyboardInterrupt:
            print("\n\nStopping test...")
            break
        except Exception as e:
            print(f"❌ Error during test: {e}")
            break
    
    # Cleanup
    hand_controller.close()
    print("Test complete!")

if __name__ == "__main__":
    main()
