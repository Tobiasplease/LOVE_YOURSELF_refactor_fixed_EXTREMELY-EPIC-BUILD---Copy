#!/usr/bin/env python3
"""
Quick Test - Just check if hand controller works
===============================================

This tests just the hand controller connection without any GUI complications.
"""

try:
    from hand_expression import HandExpressionController
    print("✅ Hand controller import successful")
    
    # Try to create a hand controller
    print("🔌 Attempting to connect to hand controller...")
    hand = HandExpressionController()
    
    if hand.serial_connection:
        print("✅ Hand controller connected successfully!")
        print("🎮 Enabling manual override...")
        hand.enable_manual_override()
        
        print("🤖 Testing hand movement...")
        # Test movement
        test_positions = [90, 90, 90, 90]  # All fingers to center
        hand.set_hand_positions(test_positions)
        print(f"📡 Sent positions: {test_positions}")
        
        # Test a small movement
        import time
        time.sleep(1)
        test_positions = [70, 110, 70, 110]  # Alternating positions
        hand.set_hand_positions(test_positions)
        print(f"📡 Sent positions: {test_positions}")
        
        time.sleep(1)
        test_positions = [90, 90, 90, 90]  # Back to center
        hand.set_hand_positions(test_positions)
        print(f"📡 Sent positions: {test_positions}")
        
        print("✅ Hand controller test SUCCESSFUL!")
        print("🎯 Your hand controller is working properly!")
        
        # Clean disconnect
        if hasattr(hand, 'disconnect'):
            hand.disconnect()
        elif hasattr(hand, 'serial_connection'):
            hand.serial_connection.close()
        print("🔌 Disconnected cleanly")
        
    else:
        print("❌ Hand controller connection failed")
        print("📝 Check Arduino connection and COM port")
        
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("📝 Hand controller module not available")
    
except Exception as e:
    print(f"❌ Error: {e}")
    print("📝 Hand controller test failed")
    import traceback
    traceback.print_exc()

input("Press Enter to exit...")
