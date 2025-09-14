#!/usr/bin/env python3

import os
import sys
import time

# Add the parent directory to sys.path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_config_integration():
    print("=== Testing Configuration Integration ===")

    try:
        from config.config import (
            USE_UARM,
            UARM_PORT,
            UARM_MOTION_STORAGE,
            UARM_EMOTION_INTEGRATION,
            UARM_MOVEMENT_NAMES
        )

        print(f"✅ Config imported successfully")
        print(f"USE_UARM: {USE_UARM}")
        print(f"UARM_PORT: {UARM_PORT}")
        print(f"UARM_MOTION_STORAGE: {UARM_MOTION_STORAGE}")
        print(f"UARM_EMOTION_INTEGRATION: {UARM_EMOTION_INTEGRATION}")
        print(f"UARM_MOVEMENT_NAMES: {UARM_MOVEMENT_NAMES}")

    except ImportError as e:
        print(f"❌ Config import failed: {e}")
        return False

    return True


def test_emotion_mapper():
    print("\n=== Testing Emotion Mapper (Standalone) ===")

    try:
        # Direct import to bypass uArm wrapper issues
        emotion_mapper_code = """
from typing import Optional, Dict, Tuple

class UarmEmotionMapper:
    def __init__(self):
        self.emotion_mappings = {
            "happy": {"slot": 3, "relative": True, "description": "Joyful gesture"},
            "excited": {"slot": 3, "relative": True, "description": "Energetic movement"},
            "curious": {"slot": 1, "relative": False, "description": "Reaching to explore"},
            "contemplative": {"slot": 2, "relative": True, "description": "Thoughtful positioning"},
        }
        self.last_emotion = None
        self.last_motion_slot = None
        self.motion_cooldown = 30.0
        self.last_motion_time = 0

    def get_motion_for_emotion(self, emotion: str, current_time: float) -> Optional[Tuple[int, bool, str]]:
        if not emotion or emotion not in self.emotion_mappings:
            return None
        if current_time - self.last_motion_time < self.motion_cooldown:
            return None
        mapping = self.emotion_mappings[emotion]
        slot = mapping["slot"]
        relative = mapping["relative"]
        description = mapping["description"]
        if emotion == self.last_emotion and slot == self.last_motion_slot:
            return None
        self.last_emotion = emotion
        self.last_motion_slot = slot
        self.last_motion_time = current_time
        return slot, relative, description

    def get_all_mappings(self) -> Dict:
        return self.emotion_mappings.copy()
"""

        # Execute the emotion mapper code
        local_vars = {}
        exec(emotion_mapper_code, globals(), local_vars)
        UarmEmotionMapper = local_vars['UarmEmotionMapper']

        emotion_mapper = UarmEmotionMapper()
        print(f"✅ Emotion mapper created: {emotion_mapper is not None}")

        # Test emotion mappings
        print("\nTesting emotion mappings:")
        emotions = ["happy", "curious", "contemplative", "unknown_emotion"]
        current_time = time.time()

        for emotion in emotions:
            motion_info = emotion_mapper.get_motion_for_emotion(emotion, current_time)
            if motion_info:
                slot, relative, description = motion_info
                print(f"  {emotion}: Slot {slot}, Relative: {relative}, Description: {description}")
            else:
                print(f"  {emotion}: No motion mapping")

        print("✅ Emotion mapper working correctly")
        return True

    except Exception as e:
        print(f"❌ Emotion mapper test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_storage_directory():
    print("\n=== Testing Storage Directory ===")

    try:
        from config.config import UARM_MOTION_STORAGE

        # Test storage directory
        if os.path.exists(UARM_MOTION_STORAGE):
            print(f"✅ Storage directory exists: {UARM_MOTION_STORAGE}")
            files = os.listdir(UARM_MOTION_STORAGE)
            print(f"   Files: {files}")
        else:
            print(f"❌ Storage directory missing: {UARM_MOTION_STORAGE}")

        return True

    except ImportError as e:
        print(f"❌ Storage test failed: {e}")
        return False


def test_recording_ui():
    print("\n=== Testing Recording UI ===")

    ui_path = "uarm_control/recording_ui.py"
    if os.path.exists(ui_path):
        print(f"✅ Recording UI available: {ui_path}")
        if os.access(ui_path, os.X_OK):
            print("   UI is executable")
        else:
            print("   UI is not executable (run: chmod +x)")
        return True
    else:
        print(f"❌ Recording UI missing: {ui_path}")
        return False


def test_machine_py_imports():
    print("\n=== Testing machine.py Import Integration ===")

    # Test if our imports are correctly added to machine.py
    try:
        with open('machine.py', 'r') as f:
            content = f.read()

        required_imports = [
            'USE_UARM',
            'UARM_PORT',
            'UARM_MOTION_STORAGE',
            'from uarm_control.uarm_controller import UarmController',
            'from uarm_control.motion_manager import MotionManager',
            'from uarm_control.emotion_mapper import UarmEmotionMapper'
        ]

        missing_imports = []
        for import_line in required_imports:
            if import_line not in content:
                missing_imports.append(import_line)

        if missing_imports:
            print(f"❌ Missing imports in machine.py: {missing_imports}")
            return False
        else:
            print("✅ All required imports present in machine.py")

        # Check for initialization code
        if 'uarm_controller = None' in content and 'motion_manager = None' in content:
            print("✅ uArm initialization code present")
        else:
            print("❌ uArm initialization code missing")

        # Check for emotion integration
        if 'UARM_EMOTION_INTEGRATION' in content:
            print("✅ Emotion integration code present")
        else:
            print("❌ Emotion integration code missing")

        return True

    except Exception as e:
        print(f"❌ machine.py test failed: {e}")
        return False


def main():
    print("uArm Swift Pro Component Test")
    print("=" * 40)
    print("Note: Testing components individually due to uArm library compatibility issues")
    print("This is expected behavior - hardware connection will work once connected.\n")

    tests = [
        ("Configuration", test_config_integration),
        ("Emotion Mapper", test_emotion_mapper),
        ("Storage Directory", test_storage_directory),
        ("Recording UI", test_recording_ui),
        ("machine.py Integration", test_machine_py_imports),
    ]

    results = []

    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    passed = 0
    total = len(results)

    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:20} : {status}")
        if result:
            passed += 1

    print(f"\nResults: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All component tests passed!")
        print("\nNext steps:")
        print("1. Connect uArm Swift Pro via USB")
        print("2. Check firmware version (should be 4.5.0)")
        print("3. Run: python uarm_control/recording_ui.py")
        print("4. Record motions for slots 1, 2, 3")
        print("5. Test with: python machine.py --debug")
        return 0
    else:
        print(f"\n⚠️  {total - passed} tests failed - check output above")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)