#!/usr/bin/env python3
"""
Essential Emotional System - Full Functionality Runner
====================================================

This launches the complete emotional hand system with ALL essential controls preserved.

Features you requested:
✅ Manual control for recording emotions (essential for training)
✅ Physics toggle - uncheck for direct/responsive mode (essential)
✅ Larger canvas (800x500) for better control precision
✅ Direction reversing controls
✅ Baseline position and gravity controls  
✅ Wave characteristics controls
✅ Preset emotions matching main system (energized, alert, calm, etc.)
✅ All physics parameters preserved and accessible
✅ Learning system integration that actually works

Usage:
1. Run this script
2. Connect hand controller
3. Select emotion and enable "Manual Mouse Control"
4. Disable "Physics Mode" for full responsiveness (recommended for recording)
5. Move mouse in canvas to record emotion
6. Click "Start Learning" to record, then "Stop Learning" when done
7. Select learned emotion and click "Express Emotion" to apply it

Author: Essential Emotional AI Team
"""

import sys
import os

# Add the servo_control directory to the path so imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the main system
try:
    from essential_emotional_system_full import EmotionalMovementSystem
    
    if __name__ == "__main__":
        print("🎭 Starting Essential Emotional Movement System - FULL FUNCTIONALITY")
        print("=" * 70)
        print("✅ All essential controls preserved")
        print("✅ Manual mouse control for recording")
        print("✅ Physics toggle (uncheck for responsive mode)")
        print("✅ Larger canvas for better precision")
        print("✅ All wave/gravity/physics parameters")
        print("✅ Preset emotions from main system")
        print("=" * 70)
        print()
        
        # Create and run the system
        app = EmotionalMovementSystem()
        app.run()
        
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure essential_conscious_cursor.py and movement_learning.py are available")
    print("Current directory:", os.getcwd())
    print("Script directory:", os.path.dirname(os.path.abspath(__file__)))
except Exception as e:
    print(f"❌ Error starting system: {e}")
    input("Press Enter to exit...")
