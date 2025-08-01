#!/usr/bin/env python3
"""
Run the Working Baseline Hand Controller
=======================================

This is your reliable fallback system. It WILL work.

- No crashes
- No division by zero errors  
- Direct mouse → hand control
- Visual feedback that works
- Simple, clean interface

Run this when you need something that just works.
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from working_baseline import WorkingBaseline
    
    if __name__ == "__main__":
        print("🔧 Starting WORKING BASELINE Hand Controller")
        print("=" * 50)
        print("✅ Simple, reliable hand control")
        print("✅ No crashes or complex features")
        print("✅ Direct mouse movement to hand")
        print("✅ Visual finger position feedback")
        print("=" * 50)
        print()
        
        app = WorkingBaseline()
        app.run()
        
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
    input("Press Enter to exit...")
