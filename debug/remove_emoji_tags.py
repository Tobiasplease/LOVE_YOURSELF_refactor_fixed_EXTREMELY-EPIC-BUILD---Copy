#!/usr/bin/env python3
"""Remove all [EMOJI] tags from the codebase"""
import os
import re

def remove_emoji_tags_from_file(filepath):
    """Remove [EMOJI] tags from a single file"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_count = content.count('[EMOJI]')
        if original_count == 0:
            return 0
            
        # Remove [EMOJI] tags
        new_content = content.replace('[EMOJI]', '')
        
        # Clean up any double spaces that might result
        new_content = re.sub(r'  +', ' ', new_content)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(new_content)
            
        print(f"Removed {original_count} [EMOJI] tags from {filepath}")
        return original_count
        
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return 0

def remove_emoji_tags_from_directory():
    """Remove [EMOJI] tags from all Python files in the project"""
    total_removed = 0
    processed_files = 0
    
    # Focus on specific directories to avoid issues
    target_dirs = [
        'hand_control',
        'captioner', 
        'mood',
        'drawing',
        'event_logging',
        'utils',
        'perception',
        'breathing'
    ]
    
    # Also check root files
    root_files = ['machine.py']
    
    for filename in root_files:
        if os.path.exists(filename):
            count = remove_emoji_tags_from_file(filename)
            total_removed += count
            if count > 0:
                processed_files += 1
    
    for target_dir in target_dirs:
        if not os.path.exists(target_dir):
            continue
            
        for root, dirs, files in os.walk(target_dir):
            for file in files:
                if file.endswith('.py'):
                    filepath = os.path.join(root, file)
                    count = remove_emoji_tags_from_file(filepath)
                    total_removed += count
                    if count > 0:
                        processed_files += 1
    
    print(f"\nSUMMARY: Removed {total_removed} [EMOJI] tags from {processed_files} files")

if __name__ == "__main__":
    print("Removing all [EMOJI] tags from the codebase...")
    remove_emoji_tags_from_directory()
    print("Done!")