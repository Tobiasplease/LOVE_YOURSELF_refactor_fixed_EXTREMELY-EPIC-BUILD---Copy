#!/usr/bin/env python3
"""
Clear all stored memory/motifs/beliefs to give the system a fresh start.
"""

import os
import shutil
import sys

# Add the parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import MOOD_SNAPSHOT_FOLDER

def clear_memory():
    """Clear all memory files and cached data."""
    print("Clearing system memory...")
    
    # Reset global person recognition system
    try:
        from captioner.person_recognition import reset_person_recognition
        reset_person_recognition()
        print("Reset person recognition system")
    except Exception as e:
        print(f"Could not reset person recognition: {e}")
    
    if os.path.exists(MOOD_SNAPSHOT_FOLDER):
        # Clear memory files but keep folder structure
        files_cleared = []
        
        # Common memory file patterns
        memory_patterns = [
            "beliefs.json", 
            "motifs.json", 
            "memory.json",
            "long_memory.json",
            "last_session.txt",
            "session_*.json"
        ]
        
        for root, dirs, files in os.walk(MOOD_SNAPSHOT_FOLDER):
            for file in files:
                file_path = os.path.join(root, file)
                
                # Clear specific memory files
                for pattern in memory_patterns:
                    if pattern.replace("*", "") in file or file.endswith('.json'):
                        try:
                            os.remove(file_path)
                            files_cleared.append(file_path)
                        except Exception as e:
                            print(f"Could not remove {file_path}: {e}")
        
        print(f"Cleared {len(files_cleared)} memory files:")
        for file_path in files_cleared:
            print(f"  - {os.path.basename(file_path)}")
    else:
        print(f"Memory folder {MOOD_SNAPSHOT_FOLDER} does not exist")
    
    print("\nMemory cleared! System will start fresh on next run.")
    print("Note: YOLO object detection has also been disabled in machine.py")

if __name__ == "__main__":
    clear_memory()