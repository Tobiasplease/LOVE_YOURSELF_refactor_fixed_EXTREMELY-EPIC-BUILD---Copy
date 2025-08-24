#!/usr/bin/env python3
"""
Clear corrupted beliefs from memory system.
Run this to reset beliefs about non-existent objects like laptops and isolation themes.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import glob
import json
from config.config import MOOD_SNAPSHOT_FOLDER

def clear_corrupted_beliefs():
    """Clear belief-related data files to reset memory system."""
    
    # Patterns that indicate corrupted beliefs
    corrupted_patterns = [
        "laptop", "isolation", "loneliness", "deafening",
        "dimly lit room", "quiet in the room"
    ]
    
    print("Clearing corrupted memory data...")
    
    # Find and clear memory-related files
    memory_files = glob.glob(os.path.join(MOOD_SNAPSHOT_FOLDER, "*.json"))
    
    cleared_count = 0
    for file_path in memory_files:
        if any(pattern in file_path.lower() for pattern in corrupted_patterns):
            try:
                os.remove(file_path)
                print(f"Removed: {os.path.basename(file_path)}")
                cleared_count += 1
            except Exception as e:
                print(f"Error removing {file_path}: {e}")
    
    print(f"Cleared {cleared_count} corrupted memory files")
    print("Memory system reset. Fresh observations should no longer be contaminated.")

if __name__ == "__main__":
    clear_corrupted_beliefs()