#!/usr/bin/env python3
"""
Quick script to show the most recent drawing prompt for debugging.
"""

import json
import os
import sys
import glob
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import MOOD_SNAPSHOT_FOLDER

def show_latest_drawing_prompt():
    """Show the most recent drawing prompt from logs."""
    
    # Find all log files
    log_pattern = os.path.join(MOOD_SNAPSHOT_FOLDER, "*-event-log.json")
    log_files = glob.glob(log_pattern)
    
    # Also check for all-run-log.json
    all_run_log = os.path.join(MOOD_SNAPSHOT_FOLDER, "all-run-log.json")
    if os.path.exists(all_run_log):
        log_files.append(all_run_log)
    
    if not log_files:
        print(f"No log files found in {MOOD_SNAPSHOT_FOLDER}!")
        return
    
    # Sort by modification time, most recent first
    log_files.sort(key=os.path.getmtime, reverse=True)
    
    print("Searching for most recent drawing prompt...")
    
    # Search through recent log files
    for log_file in log_files[:3]:  # Check last 3 runs
        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            # Handle both single entries and lists
            entries = data if isinstance(data, list) else [data]
                
            # Process entries in reverse (most recent first)
            for entry in reversed(entries):
                try:
                    
                    # Look for drawing decision with prompt
                    if (entry.get('type') == 'DECISION' and 
                        entry.get('data', {}).get('decision') == 'trigger_drawing'):
                        
                        data = entry['data']
                        drawing_prompt = data.get('drawing_prompt', 'No prompt found')
                        reflection = data.get('reflection', 'No reflection')
                        mood = data.get('mood', 'Unknown')
                        
                        print("=" * 80)
                        print("MOST RECENT DRAWING PROMPT:")
                        print("=" * 80)
                        print(f"Time: {entry.get('timestamp', 'Unknown')}")
                        print(f"Mood: {mood}")
                        print()
                        print("Drawing Prompt:")
                        print(f"   {drawing_prompt}")
                        print()
                        print("Reflection Context:")
                        print(f"   {reflection}")
                        print("=" * 80)
                        return
                        
                except Exception:
                    continue
                    
        except Exception as e:
            print(f"Error reading {log_file}: {e}")
            continue
    
    print("No recent drawing prompts found in logs")

if __name__ == "__main__":
    show_latest_drawing_prompt()