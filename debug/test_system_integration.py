#!/usr/bin/env python3
"""
Quick test to verify the temporal emotional system works in the main application context.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mood.mood_factory import create_mood_engine
from config.config import USE_EXPERIENTIAL_MOOD, DEBUG_TEMPORAL_EMOTIONS

def test_main_system_integration():
    """Test that the main system creates the correct mood engine"""
    print("=== Testing Main System Integration ===")
    print(f"USE_EXPERIENTIAL_MOOD = {USE_EXPERIENTIAL_MOOD}")
    print(f"DEBUG_TEMPORAL_EMOTIONS = {DEBUG_TEMPORAL_EMOTIONS}")
    
    # This is exactly what machine.py does
    mood_engine = create_mood_engine()
    
    print(f"Created mood engine type: {type(mood_engine).__name__}")
    
    # Test that it has temporal capabilities
    if hasattr(mood_engine, 'temporal_engine'):
        print("PASS: Temporal emotional system is active")
        
        # Test a basic mood analysis
        mood = mood_engine.analyze_mood("I see a book on the table")
        print(f"PASS: Mood analysis works: {mood:.3f}")
        
        # Test temporal context (if available)
        if hasattr(mood_engine, 'get_temporal_prompt_context'):
            context = mood_engine.get_temporal_prompt_context()
            print(f"PASS: Temporal context available: '{context}'")
        
        print("PASS: System is ready - temporal emotional system is active by default!")
        return True
    else:
        print("ERROR: Temporal emotional system not found")
        return False

if __name__ == "__main__":
    success = test_main_system_integration()
    if success:
        print("\nSUCCESS: The temporal emotional system is enabled by default!")
        print("Just run: python machine.py")
    else:
        print("\nFAILURE: System integration issue")
        sys.exit(1)