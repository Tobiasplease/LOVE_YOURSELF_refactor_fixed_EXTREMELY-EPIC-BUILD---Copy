#!/usr/bin/env python3
"""
Quick Test - Physics Hand Interface

Run this tomorrow to test the physics-based hand control concept.
Simply move your mouse around the black canvas to control the hand!

Usage: python debug/test_physics_hand_interface.py
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from debug.hand_control_interface import PhysicsHandInterface

def main():
    print("=" * 60)
    print("🎮 PHYSICS-BASED HAND CONTROL INTERFACE")
    print("=" * 60)
    print()
    print("💡 CONCEPT:")
    print("   - Move mouse over black canvas to control finger positions")
    print("   - Adjust physics sliders to change spring/damping behavior") 
    print("   - Record movements and save as presets")
    print("   - Real-time physics simulation with visual feedback")
    print()
    print("🎯 TOMORROW'S EXPERIMENT:")
    print("   - Connect to hand servos and 'conduct' movements by hand")
    print("   - Find physics parameters that feel natural")
    print("   - Record successful gesture patterns")
    print("   - Export parameters back to main system")
    print()
    print("🚀 Starting interface...")
    print()
    
    try:
        interface = PhysicsHandInterface()
        interface.run()
    except KeyboardInterrupt:
        print("\n👋 Interface closed by user")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
