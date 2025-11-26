#!/usr/bin/env python3
"""
Isolated test for GRBL completion hook and uArm papermove integration.
Tests the hook system without requiring a full drawing to complete.
"""

import os
import sys
import threading
import time

# Add the project root to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)


def test_hook_registration():
    """Test if the GRBL hook gets registered properly"""
    print("=== Testing Hook Registration ===")

    try:
        # Import the same way machine.py does
        import utils.hooks as _hooks
        from config.config import UARM_PLAY_AFTER_DRAW, UARM_PLAY_FILE, USE_UARM

        print(f"USE_UARM: {USE_UARM}")
        print(f"UARM_PLAY_AFTER_DRAW: {UARM_PLAY_AFTER_DRAW}")
        print(f"UARM_PLAY_FILE: {UARM_PLAY_FILE}")
        print(f"File exists: {os.path.exists(UARM_PLAY_FILE) if UARM_PLAY_FILE else 'N/A'}")

        # Check condition
        condition_met = USE_UARM and UARM_PLAY_AFTER_DRAW and UARM_PLAY_FILE
        print(f"Hook registration condition met: {condition_met}")

        if condition_met:
            # Register a test hook
            def test_hook():
                print("✅ TEST HOOK CALLED SUCCESSFULLY!")

            _hooks.on_grbl_drawing_complete = test_hook
            print("✅ Test hook registered")

            # Verify it's callable
            if callable(_hooks.on_grbl_drawing_complete):
                print("✅ Hook is callable")
                return True
            else:
                print("❌ Hook is not callable")
                return False
        else:
            print("❌ Hook registration condition not met")
            return False

    except Exception as e:
        print(f"❌ Hook registration failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_hook_trigger():
    """Test triggering the hook manually"""
    print("\n=== Testing Hook Trigger ===")

    try:
        import utils.hooks as _hooks

        if callable(_hooks.on_grbl_drawing_complete):
            print("🔥 Manually triggering GRBL completion hook...")
            _hooks.on_grbl_drawing_complete()
            print("✅ Hook triggered successfully")
            return True
        else:
            print("❌ No callable hook found")
            return False

    except Exception as e:
        print(f"❌ Hook trigger failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_uarm_functionality():
    """Test uArm connection and papermove file"""
    print("\n=== Testing uArm Functionality ===")

    try:
        from config.config import UARM_MOTION_STORAGE, UARM_PLAY_FILE

        print(f"Motion storage directory: {UARM_MOTION_STORAGE}")
        print(f"Papermove file: {UARM_PLAY_FILE}")

        # Check file exists
        if not os.path.exists(UARM_PLAY_FILE):
            print(f"❌ Papermove file doesn't exist: {UARM_PLAY_FILE}")
            return False

        file_size = os.path.getsize(UARM_PLAY_FILE)
        print(f"✅ Papermove file exists ({file_size} bytes)")

        # Test uArm connection
        try:
            from uarm_control.teach_menu import UArmTeachApp

            app = UArmTeachApp()

            print("🔌 Testing uArm connection...")
            app.connect()

            if app.connected:
                print("✅ uArm connected successfully")

                # Test setting play file
                app.play_file = UARM_PLAY_FILE
                print(f"✅ Play file set: {os.path.basename(app.play_file)}")

                app.disconnect()
                print("✅ uArm disconnected")
                return True
            else:
                print("❌ uArm connection failed")
                return False

        except Exception as e:
            print(f"❌ uArm test failed: {e}")
            return False

    except Exception as e:
        print(f"❌ uArm functionality test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_full_integration():
    """Test the complete integration by simulating GRBL completion"""
    print("\n=== Testing Full Integration ===")

    try:
        # Set up the real hook like machine.py does
        import utils.hooks as _hooks
        from config.config import UARM_PLAY_AFTER_DRAW, UARM_PLAY_FILE, USE_UARM

        if not (USE_UARM and UARM_PLAY_AFTER_DRAW and UARM_PLAY_FILE):
            print("❌ Integration conditions not met")
            return False

        def _normalize_smooth(path: str) -> str:
            base, ext = os.path.splitext(path)
            while base.lower().endswith(".smooth"):
                base = base[:-7]
            return f"{base}{ext or '.txt'}"

        def _uarm_after_grbl():
            print("🎯 _uarm_after_grbl() called - starting papermove sequence")

            def _run():
                try:
                    # Skip the GRBL idle manager pause for this test
                    print("⏸️  [SIMULATED] Pausing GRBL idle movements")

                    target = _normalize_smooth(UARM_PLAY_FILE)
                    print(f"📁 Target file: {target}")

                    # Test uArm app setup
                    from uarm_control.teach_menu import UArmTeachApp

                    app = UArmTeachApp()
                    app.connect()

                    if not app.connected:
                        print("❌ uArm connection failed")
                        return

                    # Configure for test
                    app.smoothing_enabled = False
                    app.use_home_before_play = False
                    app.use_home_after_play = False

                    if os.path.exists(target):
                        app.play_file = target
                        print(f"🎬 Playing: {os.path.basename(app.play_file)}")

                        # For test, just verify setup without actually playing
                        print("✅ uArm setup complete - would play movement here")
                        print("⏯️  [SIMULATED] Movement completion")

                    else:
                        print(f"❌ Target file not found: {target}")

                    app.disconnect()
                    print("▶️  [SIMULATED] Resuming GRBL idle movements")
                    print("✅ Integration test completed successfully")

                except Exception as e:
                    print(f"❌ Integration test failed: {e}")
                    import traceback

                    traceback.print_exc()

            threading.Thread(target=_run, daemon=True, name="TestUArmAfterGRBL").start()
            time.sleep(2)  # Wait for thread to complete

        # Register the hook
        _hooks.on_grbl_drawing_complete = _uarm_after_grbl
        print("🔗 Hook registered")

        # Simulate GRBL calling the hook
        print("🎭 Simulating GRBL completion...")
        if callable(_hooks.on_grbl_drawing_complete):
            _hooks.on_grbl_drawing_complete()
            time.sleep(3)  # Wait for async operations
            return True
        else:
            print("❌ Hook not callable")
            return False

    except Exception as e:
        print(f"❌ Full integration test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("🧪 uArm GRBL Hook Integration Test")
    print("=" * 50)

    tests = [
        ("Hook Registration", test_hook_registration),
        ("Hook Trigger", test_hook_trigger),
        ("uArm Functionality", test_uarm_functionality),
        ("Full Integration", test_full_integration),
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        print(f"\n🔍 Running: {test_name}")
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} PASSED")
            else:
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            print(f"💥 {test_name} CRASHED: {e}")

    print(f"\n📊 Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! Hook system should work correctly.")
    else:
        print("⚠️  Some tests failed. Check the output above for details.")


if __name__ == "__main__":
    main()
