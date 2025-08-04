#!/usr/bin/env python3
"""
Script to remove only test files from servo control directory.
Safe cleanup - removes only test_*.py files.
"""
import os
import shutil
from datetime import datetime

def cleanup_test_files():
    """Remove only test files safely."""
    
    print("🧹 Starting test file cleanup...")
    print(f"📁 Working directory: {os.getcwd()}")
    
    # Get all files in current directory
    all_files = [f for f in os.listdir('.') if os.path.isfile(f)]
    print(f"📊 Found {len(all_files)} total files")
    
    # Find test files
    test_files = [f for f in all_files if f.startswith('test_') and f.endswith('.py')]
    
    if not test_files:
        print("✅ No test files found to remove!")
        return 0
    
    # Create backup directory
    backup_dir = f"DELETED_TEST_FILES_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    if not os.path.exists(backup_dir):
        os.makedirs(backup_dir)
        print(f"📦 Created backup directory: {backup_dir}")
    
    print(f"\n🗑️ Test files to remove: {len(test_files)}")
    for f in sorted(test_files):
        print(f"   🧪 {f}")
    
    # Move test files to backup
    deleted_count = 0
    for file in test_files:
        try:
            backup_path = os.path.join(backup_dir, file)
            shutil.move(file, backup_path)
            deleted_count += 1
            print(f"🗑️ Moved: {file}")
        except Exception as e:
            print(f"❌ Error moving {file}: {e}")
    
    print(f"\n✅ TEST FILE CLEANUP COMPLETE!")
    print(f"🗑️ Moved {deleted_count} test files to {backup_dir}")
    print(f"📁 Remaining files: {len(all_files) - deleted_count}")
    
    return deleted_count

if __name__ == "__main__":
    cleanup_test_files()
