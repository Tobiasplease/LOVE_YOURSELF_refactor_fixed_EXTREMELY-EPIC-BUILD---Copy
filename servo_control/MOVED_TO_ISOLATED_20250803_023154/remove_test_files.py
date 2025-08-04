#!/usr/bin/env python3
"""
Safe cleanup script to remove only test files from servo control directory.
"""
import os
import shutil
from datetime import datetime

def cleanup_test_files():
    """Remove only test files, keeping all essential code."""
    
    print("🧹 Starting test file cleanup...")
    print(f"📁 Working directory: {os.getcwd()}")
    
    # Get all files in current directory
    all_files = [f for f in os.listdir('.') if os.path.isfile(f)]
    print(f"📊 Found {len(all_files)} total files")
    
    # Find test files (files starting with 'test_')
    test_files = [f for f in all_files if f.startswith('test_')]
    
    if not test_files:
        print("✅ No test files found to remove!")
        return 0
    
    # Create backup directory for test files
    backup_dir = f"TEST_FILES_BACKUP_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    if test_files:
        os.makedirs(backup_dir)
        print(f"📦 Created backup directory: {backup_dir}")
    
    print(f"\n📋 TEST FILES TO REMOVE:")
    for f in sorted(test_files):
        print(f"   🧪 {f}")
    
    print(f"\n⚠️ About to remove {len(test_files)} test files.")
    print("These will be moved to backup directory first for safety.")
    
    # Move test files to backup
    moved_count = 0
    for file in test_files:
        try:
            backup_path = os.path.join(backup_dir, file)
            shutil.move(file, backup_path)
            moved_count += 1
            print(f"🗑️ Moved: {file}")
        except Exception as e:
            print(f"❌ Error moving {file}: {e}")
    
    print(f"\n✅ TEST FILE CLEANUP COMPLETE!")
    print(f"🗑️ Moved {moved_count} test files to {backup_dir}")
    print(f"✅ All essential code files preserved")
    
    # Show remaining files
    remaining_files = [f for f in os.listdir('.') if os.path.isfile(f) and not f.startswith('TEST_FILES_BACKUP')]
    print(f"\n📁 Remaining files ({len(remaining_files)}):")
    for f in sorted(remaining_files):
        print(f"   📄 {f}")
    
    return moved_count

if __name__ == "__main__":
    cleanup_test_files()
