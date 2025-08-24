#!/usr/bin/env python3
"""
Force a complete memory reset by clearing all state files and cached data.
Run this script before starting the system for a truly fresh start.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import glob


def force_memory_reset():
    """Force complete memory reset"""
    print("Forcing Complete Memory Reset...")
    print("=" * 50)
    
    # Remove state files
    state_files = [
        "event_log/system_state.json",
        "event_log/lifetime_state.json", 
        "event_log/last_session.txt"
    ]
    
    removed_count = 0
    for file_path in state_files:
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"Removed: {file_path}")
            removed_count += 1
        else:
            print(f"Not found: {file_path}")
    
    # Clear any cached extractors/analyzers
    try:
        from utils.simple_motif_extractor_fixed import get_simple_motif_extractor
        extractor = get_simple_motif_extractor()
        extractor.clear_history()
        print("Cleared motif extractor history")
    except Exception as e:
        print(f"Note: Could not clear motif extractor: {e}")
    
    # Clear thematic analyzer if it exists
    try:
        from utils.thematic_analyzer import get_thematic_analyzer
        analyzer = get_thematic_analyzer(interval=1)
        if hasattr(analyzer, 'clear'):
            analyzer.clear()
            print("Cleared thematic analyzer")
    except Exception as e:
        print(f"Note: Could not clear thematic analyzer: {e}")
    
    print("\n" + "=" * 50)
    if removed_count > 0:
        print(f"SUCCESS: Removed {removed_count} state files")
        print("Memory completely reset - next startup will be truly fresh!")
        print("\nExpected behavior on next startup:")
        print("- No 'remembered' motifs or objects")
        print("- Fresh emotional associations") 
        print("- Clean awakening without references to past sessions")
        print("- New motif extraction using working system")
    else:
        print("INFO: No state files found to remove (already clean)")
    
    print("=" * 50)


if __name__ == "__main__":
    force_memory_reset()