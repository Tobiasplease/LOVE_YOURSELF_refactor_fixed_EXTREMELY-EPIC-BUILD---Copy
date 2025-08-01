#!/usr/bin/env python3
"""
🎭 ESSENTIAL EMOTIONAL MOVEMENT SYSTEM 🎭
=========================================

This is what we've been building - the CORE of emotional hand expression!

WHAT THIS DOES:
✨ Record your movements while expressing emotions
✨ AI learns YOUR unique emotional patterns  
✨ AI cursor expresses learned emotions autonomously
✨ Hand moves based on AI's emotional state
✨ Real-time transitions between emotions

HOW TO USE:
1. Run this script
2. Connect hand controller (optional - works without)
3. Type emotion name (happy, sad, excited, etc.)
4. Click "Start Learning" and move mouse to express that emotion
5. Click "Stop Learning" when done
6. Select the emotion and click "Express This Emotion"
7. Watch the AI express YOUR movement patterns!

This is the clean, streamlined version that preserves the revolutionary core
while removing all the complexity that was hiding the magic.

Let's do this! 🚀
"""

import os
import sys

def main():
    print("🎭" + "="*60 + "🎭")
    print("   ESSENTIAL EMOTIONAL MOVEMENT SYSTEM")
    print("   Pure emotional hand control - the core magic!")
    print("🎭" + "="*60 + "🎭")
    print()
    print("🚀 Starting the essential system...")
    print()
    
    # Make sure we're in the right directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    try:
        # Import and run the essential system
        from essential_emotional_system import EmotionalMovementSystem
        
        print("✅ Core systems loaded successfully!")
        print()
        print("💡 QUICK START:")
        print("   1. Type an emotion name (happy, sad, excited, etc.)")
        print("   2. Click 'Start Learning'")
        print("   3. Move your mouse to express that emotion")
        print("   4. Click 'Stop Learning' when done")
        print("   5. Select the emotion and click 'Express This Emotion'")
        print("   6. Watch the AI express YOUR patterns!")
        print()
        print("🎭 Launching interface...")
        print()
        
        # Create and run the system
        app = EmotionalMovementSystem()
        app.run()
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print()
        print("🔧 Make sure these files exist in the same directory:")
        print("   - essential_emotional_system.py")
        print("   - essential_conscious_cursor.py") 
        print("   - movement_learning.py")
        print("   - hand_expression.py (optional, for physical hand)")
        print()
        
    except KeyboardInterrupt:
        print()
        print("🛑 System stopped by user")
        
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        print()
        print("🔧 Check that all required files are present and Python modules are available")
    
    print()
    print("🎭 Thanks for using the Essential Emotional Movement System!")
    print("   Keep the magic alive! ✨")


if __name__ == "__main__":
    main()
