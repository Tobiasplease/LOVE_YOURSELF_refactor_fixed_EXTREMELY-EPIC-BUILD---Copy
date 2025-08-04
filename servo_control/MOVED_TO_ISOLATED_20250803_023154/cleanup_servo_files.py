#!/usr/bin/env python3
"""
Cleanup script to remove outdated/broken servo control files.
Keeps only the essential working versions.
"""
import os
import shutil
from datetime import datetime

def cleanup_servo_control():
    """Remove broken/outdated servo control interface files."""
    
    # Files to KEEP (essential working versions)
    keep_files = {
        'conscious_cursor_interface_PURE_MARKOV_GOLDEN_MASTER.py',  # User's identified working version
        'conscious_cursor_interface_MAIN_SERVOMARKOV_NOT_WORKING.py',  # The fixed version we're working on
        'conscious_cursor_interface_PURE_MARKOV_INFINITE_GEN_STABLE.py',  # Backup stable version
        
        # Essential support files
        'hand_expression.py',
        'markov_movement.py', 
        'movement_learning.py',
        'servo_control.py',
        'emotional_hand_controller.py',
        'debug_movement_system.py',
        'run_essential_system.py',
        'run_essential_full_system.py',
        'working_baseline.py',
        '__init__.py',
        
        # Documentation and utilities
        'fix_orphaned_code.py',  # Our cleanup script
        'README_ESSENTIAL_SYSTEM.md',
        'DYNAMIC_HAND_MOVEMENT_SYSTEM_README.md',
        'BACKUP_DOCUMENTATION_INFINITE_GENERATION.md',
        'SESSION_SUMMARY_2025-08-01.md',
        'RESTORE_INFINITE_GENERATION.bat',
        'main_function.txt',
        
        # Test files (might be useful for debugging)
        'test_dynamic_bars.py',
        'test_focus_fix.py', 
        'test_hand_only.py',
        'test_import.py',
        'test_keyboard_improvements.py',
        'test_servo_range.py',
        'test_stability_fixes.py',
        'test_visual_fixes.py'
    }
    
    # Files to DELETE (broken/intermediate versions)
    delete_patterns = [
        'conscious_cursor_interface_clean',
        'conscious_cursor_interface_WORKING',
        'conscious_cursor_interface_FIXED',
        'conscious_cursor_interface_FINAL', 
        'conscious_cursor_interface_RESTORED.py',  # Original broken version
        'conscious_cursor_interface_RESTORED_FIXED.py',  # Temp file
        'conscious_cursor_interface_RESTORED_BACKUP',
        'conscious_cursor_interface_MARKOV_CLEAN',
        'conscious_cursor_interface_MARKOV_ONLY',
        'conscious_cursor_interface_PURE_MARKOV_WORKING_INFINITE',  # Long backup name
        'conscious_cursor_interface_SIMPLE_FIX',
        'conscious_cursor_interface_CLEAN_MARKOV',
        'conscious_cursor_interface.py',  # Original version
        'conscious_cursor.py',  # Very old version
        'essential_conscious_cursor.py',
        'essential_emotional_system',
        'movement_learning_backup.py',
        'movement_learning_clean.py', 
        'hand_expression_corrupted.py',
        'hand_expression_fixed.py',
        'hand_expression_WORKING_BACKUP.py'
    ]
    
    print("🧹 Starting servo control file cleanup...")
    print(f"📁 Working directory: {os.getcwd()}")
    
    # Get all files in current directory
    all_files = [f for f in os.listdir('.') if os.path.isfile(f)]
    print(f"📊 Found {len(all_files)} total files")
    
    # Create backup directory
    backup_dir = f"DELETED_BACKUP_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    if not os.path.exists(backup_dir):
        os.makedirs(backup_dir)
        print(f"📦 Created backup directory: {backup_dir}")
    
    files_to_delete = []
    files_to_keep = []
    
    # Categorize files
    for file in all_files:
        if file in keep_files:
            files_to_keep.append(file)
        else:
            # Check if file matches any delete pattern
            should_delete = False
            for pattern in delete_patterns:
                if pattern in file:
                    should_delete = True
                    break
            
            if should_delete:
                files_to_delete.append(file)
            else:
                files_to_keep.append(file)  # Unknown files - keep them safe
    
    print(f"\n📋 CLEANUP PLAN:")
    print(f"✅ Files to KEEP: {len(files_to_keep)}")
    for f in sorted(files_to_keep):
        print(f"   ✅ {f}")
    
    print(f"\n🗑️ Files to DELETE: {len(files_to_delete)}")
    for f in sorted(files_to_delete):
        print(f"   🗑️ {f}")
    
    # Confirm deletion
    if files_to_delete:
        print(f"\n⚠️ About to delete {len(files_to_delete)} files.")
        print("These will be moved to backup directory first.")
        
        # Move files to backup (safer than permanent deletion)
        deleted_count = 0
        for file in files_to_delete:
            try:
                backup_path = os.path.join(backup_dir, file)
                shutil.move(file, backup_path)
                deleted_count += 1
                print(f"🗑️ Moved: {file}")
            except Exception as e:
                print(f"❌ Error moving {file}: {e}")
        
        print(f"\n✅ CLEANUP COMPLETE!")
        print(f"🗑️ Moved {deleted_count} files to {backup_dir}")
        print(f"✅ Kept {len(files_to_keep)} essential files")
        print(f"\n📁 Remaining files:")
        for f in sorted([f for f in os.listdir('.') if os.path.isfile(f) and not f.startswith('DELETED_BACKUP')]):
            print(f"   📄 {f}")
    else:
        print("✅ No files need to be deleted!")
    
    return deleted_count

if __name__ == "__main__":
    cleanup_servo_control()
