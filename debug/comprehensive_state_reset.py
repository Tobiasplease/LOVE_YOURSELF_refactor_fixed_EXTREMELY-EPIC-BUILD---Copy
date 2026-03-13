#!/usr/bin/env python3
"""
Comprehensive state reset - clears ALL accumulated memory/context/state data.
This goes beyond the previous reset to ensure nothing persists.
"""
import sys
import os
import json
from datetime import datetime
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import MOOD_SNAPSHOT_FOLDER

def identify_state_files():
    """Find all state/memory files."""
    state_files = {}

    # Known state files
    state_files['session_state'] = os.path.join(MOOD_SNAPSHOT_FOLDER, "system_state.json")
    state_files['lifetime_state'] = os.path.join(MOOD_SNAPSHOT_FOLDER, "lifetime_state.json")
    state_files['drawing_memory'] = os.path.join(MOOD_SNAPSHOT_FOLDER, "drawing_memory.json")

    # Check for context compression cache
    compression_cache = os.path.join(MOOD_SNAPSHOT_FOLDER, "context_compression_cache.json")
    if os.path.exists(compression_cache):
        state_files['compression_cache'] = compression_cache

    # Check for any other state-related files
    if os.path.exists(MOOD_SNAPSHOT_FOLDER):
        for filename in os.listdir(MOOD_SNAPSHOT_FOLDER):
            if 'state' in filename.lower() or 'memory' in filename.lower() or 'cache' in filename.lower():
                if filename.endswith('.json'):
                    full_path = os.path.join(MOOD_SNAPSHOT_FOLDER, filename)
                    if full_path not in state_files.values():
                        state_files[f'other_{filename}'] = full_path

    return state_files

def analyze_state_file(filepath):
    """Analyze what's stored in a state file."""
    if not os.path.exists(filepath):
        return None

    try:
        with open(filepath, 'r') as f:
            data = json.load(f)

        info = {
            'size_kb': os.path.getsize(filepath) / 1024,
            'keys': list(data.keys()) if isinstance(data, dict) else 'not_dict',
        }

        # Analyze captioner state if present
        if isinstance(data, dict) and 'captioner' in data:
            cap = data['captioner']
            info['captioner'] = {
                'beliefs': len(cap.get('beliefs', {})),
                'motifs': len(cap.get('motif_counter', {})),
                'timeline_entries': len(cap.get('timeline', [])),
                'day_stones': len(cap.get('day_stones', [])),
                'known_people': len(cap.get('known_people', {})),
                'emotional_expressions': len(cap.get('emotional_expressions', [])),
                'recent_memory': len(cap.get('recent_memory', [])),
            }

        return info
    except Exception as e:
        return {'error': str(e)}

def backup_state_files(state_files):
    """Create timestamped backups before reset."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backed_up = []

    for name, filepath in state_files.items():
        if os.path.exists(filepath):
            backup_path = f"{filepath}.backup_{timestamp}"
            try:
                import shutil
                shutil.copy2(filepath, backup_path)
                backed_up.append((name, backup_path))
                print(f"  ✓ Backed up {name} → {os.path.basename(backup_path)}")
            except Exception as e:
                print(f"  ✗ Failed to backup {name}: {e}")

    return backed_up

def reset_state_files(state_files, preserve_lifetime=False):
    """Delete or reset state files."""
    reset_count = 0

    for name, filepath in state_files.items():
        if not os.path.exists(filepath):
            continue

        # Optionally preserve lifetime stats
        if preserve_lifetime and 'lifetime' in name.lower():
            print(f"  ⊘ Skipped {name} (preserving lifetime stats)")
            continue

        try:
            os.remove(filepath)
            reset_count += 1
            print(f"  ✓ Deleted {name}")
        except Exception as e:
            print(f"  ✗ Failed to delete {name}: {e}")

    return reset_count

def main():
    print("=" * 80)
    print("COMPREHENSIVE STATE RESET")
    print("=" * 80)

    # Find all state files
    print(f"\n🔍 Scanning for state files in {MOOD_SNAPSHOT_FOLDER}...")
    state_files = identify_state_files()

    if not state_files:
        print("  No state files found.")
        return

    print(f"\n📋 Found {len(state_files)} state/memory files:")
    for name, filepath in state_files.items():
        exists = "✓" if os.path.exists(filepath) else "✗"
        print(f"  {exists} {name}: {filepath}")

    # Analyze each file
    print(f"\n📊 Analyzing state content:")
    for name, filepath in state_files.items():
        if not os.path.exists(filepath):
            continue

        info = analyze_state_file(filepath)
        if info:
            print(f"\n  {name}:")
            print(f"    Size: {info['size_kb']:.1f} KB")
            if 'captioner' in info:
                print(f"    Captioner data:")
                for key, val in info['captioner'].items():
                    print(f"      - {key}: {val}")

    # Confirm reset
    print(f"\n{'='*80}")
    print("⚠️  WARNING: This will DELETE all accumulated memory/context/state!")
    print("   - All beliefs, motifs, memories")
    print("   - Environmental understanding")
    print("   - Emotional vocabulary")
    print("   - Timeline and day stones")
    print("   - Known people")
    print("   - Drawing memory")
    print(f"{'='*80}")

    response = input("\nProceed with reset? Type 'YES' to confirm: ")

    if response.strip() != 'YES':
        print("\n❌ Reset cancelled.")
        return

    # Create backups
    print(f"\n💾 Creating backups...")
    backed_up = backup_state_files(state_files)
    print(f"   Created {len(backed_up)} backup(s)")

    # Ask about lifetime stats
    preserve = input("\nPreserve lifetime stats? (y/n): ").strip().lower() == 'y'

    # Reset
    print(f"\n🗑️  Resetting state files...")
    reset_count = reset_state_files(state_files, preserve_lifetime=preserve)

    print(f"\n{'='*80}")
    print(f"✅ RESET COMPLETE")
    print(f"   - Deleted: {reset_count} files")
    print(f"   - Backed up: {len(backed_up)} files")
    print(f"\n   Next startup will begin with fresh state.")
    print(f"   Backups saved in: {MOOD_SNAPSHOT_FOLDER}")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()
