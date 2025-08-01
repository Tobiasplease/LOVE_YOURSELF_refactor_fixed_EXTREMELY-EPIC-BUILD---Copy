import sys
import os
print(f"Python version: {sys.version}")
print(f"Current working directory: {os.getcwd()}")
print("Testing essential system import...")

try:
    from essential_emotional_system_full import EmotionalMovementSystem
    print("✅ Import successful!")
    print("Creating system...")
    app = EmotionalMovementSystem()
    print("✅ System created successfully!")
    print("Ready to run!")
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
