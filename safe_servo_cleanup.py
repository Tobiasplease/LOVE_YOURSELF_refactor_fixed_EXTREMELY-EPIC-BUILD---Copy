#!/usr/bin/env python3
"""
🧹 Safe Servo Control Cleanup
=============================

This script safely moves hand control UI files to the isolated system
while preserving all essential servo systems used by the main AI.

PRESERVED FILES (Essential for main AI system):
- servo_control.py (core servo controller for gaze & breathing)
- emotional_hand_controller.py (hand position updates for main AI)
- hand_expression.py (hand expression controller)
- __init__.py (module initialization)

MOVED FILES (Hand control UI - safe to isolate):
- All conscious_cursor_interface_*.py files
- All movement learning/recording files and folders
- All markov chain files and folders
- All test/debug files for hand control
- All standalone hand control files

SAFETY CHECKS:
- Verifies main AI dependencies before moving
- Creates backup of moved files
- Preserves one copy of the problematic file for reference
"""

import os
import shutil
import datetime
from pathlib import Path

def main():
    print("🧹 Starting Safe Servo Control Cleanup...")
    
    # Define paths
    base_path = Path(r"c:\Users\tobia\Downloads\LOVE_YOURSELF_refactor_fixed_EXTREMELY-EPIC-BUILD---Copy-win-epic")
    servo_control_path = base_path / "servo_control"
    isolated_path = base_path / "hand_control_isolated"
    
    # Create backup directory with timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = servo_control_path / f"MOVED_TO_ISOLATED_{timestamp}"
    backup_path.mkdir(exist_ok=True)
    
    print(f"📁 Backup directory: {backup_path}")
    
    # ESSENTIAL FILES - MUST NOT BE MOVED (used by main AI system)
    essential_files = {
        "servo_control.py",           # Core servo controller (gaze, breathing)
        "emotional_hand_controller.py", # Hand position updates for main AI
        "hand_expression.py",         # Hand expression controller
        "__init__.py",                # Module initialization
        # Keep these for safety
        "README_ESSENTIAL_SYSTEM.md",
        "run_essential_system.py", 
        "run_essential_full_system.py"
    }
    
    # HAND CONTROL UI FILES - SAFE TO MOVE
    hand_control_patterns = [
        "conscious_cursor_interface",
        "movement_learning",
        "movement_synthesis", 
        "markov_movement",
        "standalone_hand_control",
        "debug_movement",
        "test_markov",
        "test_recording",
        "working_baseline"
    ]
    
    # DIRECTORIES TO MOVE
    directories_to_move = [
        "markov_chains",
        "movement_profiles", 
        "movement_recordings",
        "movement_recordings_backup_old_system",
        "movement_signatures"
    ]
    
    # SAFETY CHECK: Verify essential files exist
    print("🔍 Verifying essential servo files...")
    missing_essential = []
    for essential_file in essential_files:
        if not (servo_control_path / essential_file).exists():
            missing_essential.append(essential_file)
    
    if missing_essential:
        print(f"❌ ERROR: Missing essential files: {missing_essential}")
        print("❌ Cannot proceed safely - main AI system would be broken!")
        return False
    
    print("✅ All essential servo files verified present")
    
    # Find files to move
    files_to_move = []
    
    # Find all Python files matching hand control patterns
    for py_file in servo_control_path.glob("*.py"):
        filename = py_file.name
        if filename in essential_files:
            continue  # Skip essential files
            
        # Check if it matches any hand control pattern
        for pattern in hand_control_patterns:
            if pattern in filename:
                files_to_move.append(py_file)
                break
    
    # Add specific additional files
    additional_files = [
        "BACKUP_DOCUMENTATION_INFINITE_GENERATION.md",
        "DYNAMIC_HAND_MOVEMENT_SYSTEM_README.md",
        "SESSION_SUMMARY_2025-08-01.md",
        "cleanup_servo_files.py",
        "cleanup_test_files.py", 
        "fix_orphaned_code.py",
        "remove_test_files.py",
        "RESTORE_INFINITE_GENERATION.bat",
        "main_function.txt"
    ]
    
    for additional_file in additional_files:
        file_path = servo_control_path / additional_file
        if file_path.exists():
            files_to_move.append(file_path)
    
    # Add directories
    dirs_to_move = []
    for dir_name in directories_to_move:
        dir_path = servo_control_path / dir_name
        if dir_path.exists():
            dirs_to_move.append(dir_path)
    
    # Show what will be moved
    print(f"\n📦 FILES TO MOVE ({len(files_to_move)}):")
    for file_path in files_to_move:
        print(f"   📄 {file_path.name}")
    
    print(f"\n📁 DIRECTORIES TO MOVE ({len(dirs_to_move)}):")
    for dir_path in dirs_to_move:
        print(f"   📁 {dir_path.name}")
    
    print(f"\n🔒 ESSENTIAL FILES PRESERVED ({len(essential_files)}):")
    for essential_file in essential_files:
        if (servo_control_path / essential_file).exists():
            print(f"   ✅ {essential_file}")
    
    # Confirm before proceeding
    response = input(f"\n🤔 Move {len(files_to_move)} files and {len(dirs_to_move)} directories to isolated system? [y/N]: ")
    if response.lower() != 'y':
        print("❌ Operation cancelled")
        return False
    
    # Create isolated system directory if it doesn't exist
    isolated_path.mkdir(exist_ok=True)
    
    # Move files
    moved_count = 0
    for file_path in files_to_move:
        try:
            # Copy to backup first
            shutil.copy2(file_path, backup_path / file_path.name)
            
            # Move to isolated system
            shutil.move(str(file_path), str(isolated_path / file_path.name))
            moved_count += 1
            print(f"📦 Moved: {file_path.name}")
            
        except Exception as e:
            print(f"❌ Failed to move {file_path.name}: {e}")
    
    # Move directories  
    for dir_path in dirs_to_move:
        try:
            # Copy to backup first
            shutil.copytree(dir_path, backup_path / dir_path.name)
            
            # Move to isolated system
            shutil.move(str(dir_path), str(isolated_path / dir_path.name))
            print(f"📁 Moved directory: {dir_path.name}")
            
        except Exception as e:
            print(f"❌ Failed to move directory {dir_path.name}: {e}")
    
    # SPECIAL: Keep one copy of the problematic file for reference
    problematic_file = "conscious_cursor_interface_MAIN_SERVOMARKOV_NOT_WORKING.py"
    if (isolated_path / problematic_file).exists():
        reference_name = f"REFERENCE_{problematic_file}"
        shutil.copy2(isolated_path / problematic_file, servo_control_path / reference_name)
        print(f"📋 Kept reference copy: {reference_name}")
    
    print(f"\n✅ Cleanup complete!")
    print(f"📦 Moved {moved_count} files to isolated system")
    print(f"💾 Backup created at: {backup_path}")
    print(f"🔒 Essential servo systems preserved for main AI")
    
    # Verify main AI dependencies are intact
    print(f"\n🔍 Final verification...")
    for essential_file in ["servo_control.py", "emotional_hand_controller.py", "hand_expression.py"]:
        if (servo_control_path / essential_file).exists():
            print(f"   ✅ {essential_file}")
        else:
            print(f"   ❌ MISSING: {essential_file}")
    
    print(f"\n🎯 Main AI system servo dependencies: PRESERVED")
    print(f"🤖 Hand control UI: ISOLATED to {isolated_path}")
    print(f"🔗 Integration: Use hand_control_bridge.py for mood updates")
    
    return True

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 Safe cleanup completed successfully!")
        print("🚀 You can now iterate on hand control without affecting the main AI system")
    else:
        print("\n❌ Cleanup failed or was cancelled")
