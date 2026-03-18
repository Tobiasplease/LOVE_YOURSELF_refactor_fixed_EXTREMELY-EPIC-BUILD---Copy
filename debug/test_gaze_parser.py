#!/usr/bin/env python3
"""Test the new natural gaze parser."""

import sys
sys.path.insert(0, '/home/impostor/LOVE_YOURSELF_refactor_fixed_EXTREMELY-EPIC-BUILD---Copy')

from captioner.captioner import _parse_gaze_direction, _clean_caption_for_display

test_cases = [
    # New natural format (preferred)
    ("*glancing left* Something catches my eye...", "left", "Something catches my eye..."),
    ("*looking down* The desk surface is dusty.", "down", "The desk surface is dusty."),
    ("*eyes ahead* The wall remains unchanged.", "ahead", "The wall remains unchanged."),
    ("*turning right* A shadow moves.", "right", "A shadow moves."),
    ("*gazing up* The ceiling light flickers.", "up", "The ceiling light flickers."),
    ("*staring forward* Nothing new.", "ahead", "Nothing new."),

    # Legacy format (backward compatibility)
    ("Some thought... LOOK: left", "left", "Some thought..."),
    ("LOOK: down\nThe paper waits.", "down", "The paper waits."),

    # No gaze direction
    ("Just a thought without direction.", None, "Just a thought without direction."),

    # Natural phrases without asterisks (fallback)
    ("Looking down at my desk, the clutter seems familiar.", "down", None),  # cleanup test
    ("Glancing left, I notice movement.", "left", None),
]

print("Testing natural gaze parser...")
print("=" * 60)

passed = 0
failed = 0

for caption, expected_dir, expected_clean in test_cases:
    gaze = _parse_gaze_direction(caption)
    clean = _clean_caption_for_display(caption)

    dir_ok = gaze == expected_dir
    # For cleanup, just check it's not None if we didn't specify expected_clean
    clean_ok = expected_clean is None or (clean is not None and expected_clean in clean)

    if dir_ok:
        passed += 1
        status = "✅"
    else:
        failed += 1
        status = "❌"

    print(f"{status} Input: {caption[:50]}...")
    print(f"   Gaze: {gaze} (expected: {expected_dir})")
    if clean:
        print(f"   Clean: {clean[:50]}...")
    print()

print("=" * 60)
print(f"Passed: {passed}/{passed+failed}")
if failed > 0:
    print(f"Failed: {failed}")
